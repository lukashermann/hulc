from collections import OrderedDict
import logging
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
import torch
import torch.distributions as D
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy

from hulc.models.decoders.action_decoder import ActionDecoder
from hulc.utils.distributions import State

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


class HULC(pl.LightningModule):
    def __init__(
        self,
        perceptual_encoder: DictConfig,
        plan_proposal: DictConfig,
        plan_recognition: DictConfig,
        language_encoder: DictConfig,
        language_goal: DictConfig,
        visual_goal: DictConfig,
        action_decoder: DictConfig,
        kl_beta: float,
        kl_balancing_mix: float,
        state_recons: bool,
        state_recon_beta: float,
        lang_recons: bool,
        lang_recon_beta: float,
        lang_contrastive: bool,
        lang_contrastive_beta: float,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        distribution: DictConfig,
        val_instructions: DictConfig,
        img_lang_matching_clip: bool,
        lang_clip_beta: float,
        replan_freq: int = 30,
        lang_decoder: Optional[DictConfig] = None,
        lang_discriminator: Optional[DictConfig] = None,
        clip_proj: Optional[DictConfig] = None,
    ):
        super(HULC, self).__init__()
        self.perceptual_encoder = hydra.utils.instantiate(perceptual_encoder, device=self.device)
        self.setup_input_sizes(
            self.perceptual_encoder,
            plan_proposal,
            plan_recognition,
            visual_goal,
            action_decoder,
            distribution,
        )
        self.dist = hydra.utils.instantiate(distribution)
        self.plan_proposal = hydra.utils.instantiate(plan_proposal, dist=self.dist)
        self.plan_recognition = hydra.utils.instantiate(plan_recognition, dist=self.dist)
        self.visual_goal = hydra.utils.instantiate(visual_goal)
        self.lang_encoder = hydra.utils.instantiate(language_encoder) if language_encoder else None
        self.language_goal = hydra.utils.instantiate(language_goal) if language_goal else None
        self.action_decoder: ActionDecoder = hydra.utils.instantiate(action_decoder)
        self.img_lang_matching_clip = img_lang_matching_clip
        self.lang_clip_beta = lang_clip_beta
        if img_lang_matching_clip:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.proj_vis_lang = hydra.utils.instantiate(clip_proj)
        self.lang_decoder = None
        self.lang_discriminator = None
        if lang_decoder:
            self.lang_decoder = hydra.utils.instantiate(lang_decoder)
        if lang_discriminator:
            self.lang_discriminator = hydra.utils.instantiate(lang_discriminator)
            self.proj_vis_lang = hydra.utils.instantiate(clip_proj)

        self.kl_beta = kl_beta
        self.kl_balacing_mix = kl_balancing_mix
        self.state_recons = state_recons
        self.st_recon_beta = state_recon_beta
        self.lang_recons = lang_recons
        self.lang_recon_beta = lang_recon_beta
        self.lang_contrastive = lang_contrastive
        self.lang_contrastive_beta = lang_contrastive_beta
        self.modality_scope = "vis"
        self.optimizer_config = optimizer
        self.lr_scheduler = lr_scheduler
        # action_decoder.out_features = action_decoder.out_features
        # self.perceptual_encoder.lang_decoder.perceptual_features = self.perceptual_encoder.lang_decoder.perceptual_features
        self.save_hyperparameters()

        # for inference
        self.rollout_step_counter = 0
        self.replan_freq = replan_freq
        self.latent_goal = None
        self.plan = None

        # for clip loss
        if self.img_lang_matching_clip:
            self.encoded_lang_train: Optional[torch.Tensor] = None
            self.encoded_lang_val: Optional[torch.Tensor] = None
            self.train_lang_emb: Optional[torch.Tensor] = None
            self.lang_data_val = None
            self.task_to_id: Optional[Dict] = None
            self.val_dataset = None
            self.train_lang_task_ids: Optional[np.ndarray] = None
            self.val_lang_emb: Optional[torch.Tensor] = None
            self.val_lang_task_ids: Optional[np.ndarray] = None
            self.val_instructions = val_instructions

    @staticmethod
    def setup_input_sizes(
        perceptual_encoder,
        plan_proposal,
        plan_recognition,
        visual_goal,
        action_decoder,
        distribution,
    ):
        plan_proposal.perceptual_features = perceptual_encoder.latent_size
        plan_recognition.in_features = perceptual_encoder.latent_size
        visual_goal.in_features = perceptual_encoder.latent_size
        action_decoder.perceptual_features = perceptual_encoder.latent_size

        if distribution.dist == "discrete":
            plan_proposal.plan_features = distribution.class_size * distribution.category_size
            plan_recognition.plan_features = distribution.class_size * distribution.category_size
            action_decoder.plan_features = distribution.class_size * distribution.category_size
        elif distribution.dist == "continuous":
            plan_proposal.plan_features = distribution.plan_features
            plan_recognition.plan_features = distribution.plan_features
            action_decoder.plan_features = distribution.plan_features

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        assert isinstance(self.trainer, pl.Trainer)
        combined_loader_dict = self.trainer.datamodule.train_dataloader()  # type: ignore
        dataset_lengths = [len(combined_loader_dict[k]) for k in combined_loader_dict.keys()]
        dataset_size = max(dataset_lengths)
        if isinstance(self.trainer.limit_train_batches, int) and self.trainer.limit_train_batches != 0:
            dataset_size = self.trainer.limit_train_batches
        elif isinstance(self.trainer.limit_train_batches, float):
            # limit_train_batches is a percentage of batches
            dataset_size = int(dataset_size * self.trainer.limit_train_batches)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices  # type: ignore
        max_estimated_steps = (dataset_size // effective_batch_size) * self.trainer.max_epochs  # type: ignore

        if self.trainer.max_steps and self.trainer.max_steps < max_estimated_steps:  # type: ignore
            return self.trainer.max_steps  # type: ignore
        return max_estimated_steps

    def compute_warmup(self, num_training_steps: int, num_warmup_steps: Union[int, float]) -> Tuple[int, int]:
        if num_training_steps < 0:
            # less than 0 specifies to infer number of training steps
            num_training_steps = self.num_training_steps
        if isinstance(num_warmup_steps, float):
            # Convert float values to percentage of training steps to use as warmup
            num_warmup_steps *= num_training_steps
        num_warmup_steps = int(num_warmup_steps)
        return num_training_steps, num_warmup_steps

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer_config, params=self.parameters())
        if "num_warmup_steps" in self.lr_scheduler:
            self.lr_scheduler.num_training_steps, self.lr_scheduler.num_warmup_steps = self.compute_warmup(
                num_training_steps=self.lr_scheduler.num_training_steps,
                num_warmup_steps=self.lr_scheduler.num_warmup_steps,
            )
            rank_zero_info(f"Inferring number of training steps, set to {self.lr_scheduler.num_training_steps}")
            rank_zero_info(f"Inferring number of warmup steps from ratio, set to {self.lr_scheduler.num_warmup_steps}")
        scheduler = hydra.utils.instantiate(self.lr_scheduler, optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def lmp_train(
        self, perceptual_emb: torch.Tensor, latent_goal: torch.Tensor, train_acts: torch.Tensor, robot_obs: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.distributions.Distribution,
        torch.distributions.Distribution,
        NamedTuple,
    ]:
        # ------------Plan Proposal------------ #
        pp_state = self.plan_proposal(perceptual_emb[:, 0], latent_goal)
        pp_dist = self.dist.get_dist(pp_state)

        # ------------Plan Recognition------------ #
        pr_state, seq_feat = self.plan_recognition(perceptual_emb)
        pr_dist = self.dist.get_dist(pr_state)

        sampled_plan = pr_dist.rsample()  # sample from recognition net
        if self.dist.dist == "discrete":
            sampled_plan = torch.flatten(sampled_plan, start_dim=-2, end_dim=-1)

        action_loss = self.action_decoder.loss(
            sampled_plan, perceptual_emb, latent_goal, train_acts, robot_obs
        )  # type:  ignore
        kl_loss = self.compute_kl_loss(pp_state, pr_state)
        total_loss = action_loss + kl_loss

        return kl_loss, action_loss, total_loss, pp_dist, pr_dist, seq_feat

    def lmp_val(
        self, perceptual_emb: torch.Tensor, latent_goal: torch.Tensor, actions: torch.Tensor, robot_obs: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        NamedTuple,
    ]:
        # ------------Plan Proposal------------ #
        pp_state = self.plan_proposal(perceptual_emb[:, 0], latent_goal)  # (batch, 256) each
        pp_dist = self.dist.get_dist(pp_state)

        # ------------ Policy network ------------ #
        sampled_plan_pp = self.dist.sample_latent_plan(pp_dist)  # sample from proposal net
        action_loss_pp, sample_act_pp = self.action_decoder.loss_and_act(  # type:  ignore
            sampled_plan_pp, perceptual_emb, latent_goal, actions, robot_obs
        )

        mae_pp = torch.nn.functional.l1_loss(
            sample_act_pp[..., :-1], actions[..., :-1], reduction="none"
        )  # (batch, seq, 6)
        mae_pp = torch.mean(mae_pp, 1)  # (batch, 6)
        # gripper action
        gripper_discrete_pp = sample_act_pp[..., -1]
        gt_gripper_act = actions[..., -1]
        m = gripper_discrete_pp > 0
        gripper_discrete_pp[m] = 1
        gripper_discrete_pp[~m] = -1
        gripper_sr_pp = torch.mean((gt_gripper_act == gripper_discrete_pp).float())

        # ------------Plan Recognition------------ #
        pr_state, seq_feat = self.plan_recognition(perceptual_emb)
        pr_dist = self.dist.get_dist(pr_state)
        sampled_plan_pr = self.dist.sample_latent_plan(pr_dist)  # sample from recognition net
        action_loss_pr, sample_act_pr = self.action_decoder.loss_and_act(  # type:  ignore
            sampled_plan_pr, perceptual_emb, latent_goal, actions, robot_obs
        )
        mae_pr = torch.nn.functional.l1_loss(
            sample_act_pr[..., :-1], actions[..., :-1], reduction="none"
        )  # (batch, seq, 6)
        mae_pr = torch.mean(mae_pr, 1)  # (batch, 6)
        kl_loss = self.compute_kl_loss(pp_state, pr_state)
        # gripper action
        gripper_discrete_pr = sample_act_pr[..., -1]
        m = gripper_discrete_pr > 0
        gripper_discrete_pr[m] = 1
        gripper_discrete_pr[~m] = -1
        gripper_sr_pr = torch.mean((gt_gripper_act == gripper_discrete_pr).float())

        return (
            sampled_plan_pp,
            action_loss_pp,
            sampled_plan_pr,
            action_loss_pr,
            kl_loss,
            mae_pp,
            mae_pr,
            gripper_sr_pp,
            gripper_sr_pr,
            seq_feat,
        )

    def training_step(  # type: ignore
        self,
        batch: Dict[
            str,
            Dict,
        ],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        batch: list( batch_dataset_vision, batch_dataset_lang, ..., batch_dataset_differentModalities)
            - batch_dataset_vision: tuple( train_obs: Tensor,
                                           train_rgbs: tuple(Tensor, ),
                                           train_depths: tuple(Tensor, ),
                                           train_acts: Tensor ),
                                           info: Dict,
                                           idx: int
            - batch_dataset_lang: tuple( train_obs: Tensor,
                                         train_rgbs: tuple(Tensor, ),
                                         train_depths: tuple(Tensor, ),
                                         train_acts: Tensor,
                                         train_lang: Tensor   ),
                                         info: Dict,
                                         idx: int
        """
        kl_loss, action_loss, proprio_loss, lang_pred_loss, lang_contrastive_loss, lang_clip_loss, total_loss = (
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
        )

        encoders_dict = {}
        batch_size: Dict[str, int] = {}
        total_bs = 0
        for self.modality_scope, dataset_batch in batch.items():
            perceptual_emb = self.perceptual_encoder(
                dataset_batch["rgb_obs"], dataset_batch["depth_obs"], dataset_batch["robot_obs"]
            )
            if self.state_recons:
                proprio_loss += self.perceptual_encoder.state_reconstruction_loss()
            if "lang" in self.modality_scope:
                latent_goal = self.language_goal(dataset_batch["lang"])
            else:
                latent_goal = self.visual_goal(perceptual_emb[:, -1])
            kl, act_loss, mod_loss, pp_dist, pr_dist, seq_feat = self.lmp_train(
                perceptual_emb, latent_goal, dataset_batch["actions"], dataset_batch["state_info"]["robot_obs"]
            )
            if "lang" in self.modality_scope:
                if not torch.any(dataset_batch["use_for_aux_lang_loss"]):
                    batch_size["aux_lang"] = 1
                else:
                    batch_size["aux_lang"] = torch.sum(dataset_batch["use_for_aux_lang_loss"]).detach()  # type:ignore
                    if self.lang_recons:
                        lang_pred_loss += self.lang_regression_loss(
                            seq_feat, dataset_batch["lang"], dataset_batch["use_for_aux_lang_loss"]
                        )
                    if self.img_lang_matching_clip:
                        lang_clip_loss += self.clip_loss(seq_feat, latent_goal, dataset_batch["use_for_aux_lang_loss"])
                    if self.lang_contrastive:
                        lang_contrastive_loss += self.contrastive_lang_loss(
                            seq_feat, latent_goal, dataset_batch["use_for_aux_lang_loss"]
                        )
            encoders_dict[self.modality_scope] = [pp_dist, pr_dist]
            kl_loss += kl
            action_loss += act_loss
            total_loss += mod_loss
            batch_size[self.modality_scope] = dataset_batch["actions"].shape[0]
            total_bs += dataset_batch["actions"].shape[0]
            self.log(
                f"train/kl_loss_scaled_{self.modality_scope}",
                kl,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size[self.modality_scope],
            )
            self.log(
                f"train/action_loss_{self.modality_scope}",
                act_loss,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size[self.modality_scope],
            )
            self.log(
                f"train/total_loss_{self.modality_scope}",
                mod_loss,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size[self.modality_scope],
            )
        total_loss = total_loss / len(batch)  # divide accumulated gradients by number of datasets
        kl_loss = kl_loss / len(batch)
        action_loss = action_loss / len(batch)
        if self.state_recons:
            proprio_loss = proprio_loss / len(batch)
            total_loss = total_loss + self.st_recon_beta * proprio_loss
            self.log(
                "train/pred_proprio",
                self.st_recon_beta * proprio_loss,
                on_step=False,
                on_epoch=True,
                batch_size=total_bs,
            )
        if self.lang_recons:
            total_loss = total_loss + self.lang_recon_beta * lang_pred_loss
            self.log(
                "train/pred_lang",
                self.lang_recon_beta * lang_pred_loss,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size["aux_lang"],
                sync_dist=True,
            )
        if self.lang_contrastive:
            total_loss = total_loss + self.lang_contrastive_beta * lang_contrastive_loss
            self.log(
                "train/lang_contrastive",
                self.lang_contrastive_beta * lang_contrastive_loss,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size["aux_lang"],
                sync_dist=True,
            )
        if self.img_lang_matching_clip:
            total_loss = total_loss + self.lang_clip_beta * lang_clip_loss
            self.log(
                "train/lang_clip_loss",
                self.lang_clip_beta * lang_clip_loss,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size["aux_lang"],
                sync_dist=True,
            )
        self.log("train/kl_loss", kl_loss, on_step=False, on_epoch=True, batch_size=total_bs)
        self.log("train/action_loss", action_loss, on_step=False, on_epoch=True, batch_size=total_bs)
        self.log("train/total_loss", total_loss, on_step=False, on_epoch=True, batch_size=total_bs)
        return total_loss

    def compute_kl_loss(self, pp_state: State, pr_state: State) -> torch.Tensor:
        pp_dist = self.dist.get_dist(pp_state)  # prior
        pr_dist = self.dist.get_dist(pr_state)  # posterior
        # @fixme: do this more elegantly
        kl_lhs = D.kl_divergence(self.dist.get_dist(self.dist.detach_state(pr_state)), pp_dist).mean()
        kl_rhs = D.kl_divergence(pr_dist, self.dist.get_dist(self.dist.detach_state(pp_state))).mean()

        alpha = self.kl_balacing_mix
        kl_loss = alpha * kl_lhs + (1 - alpha) * kl_rhs
        kl_loss_scaled = kl_loss * self.kl_beta
        return kl_loss_scaled

    def set_kl_beta(self, kl_beta):
        """Set kl_beta from Callback"""
        self.kl_beta = kl_beta

    def lang_regression_loss(self, seq_vis_feat, gt_lang, use_for_aux_loss):
        assert self.lang_decoder is not None
        if use_for_aux_loss is not None:
            if not torch.any(use_for_aux_loss):
                return torch.tensor(0.0).to(self.device)
            seq_vis_feat = seq_vis_feat[use_for_aux_loss]
            gt_lang = gt_lang[use_for_aux_loss]
        lang_pred = self.lang_decoder(seq_vis_feat)
        cos_sim = ((lang_pred * gt_lang).sum(-1)) / (
            torch.linalg.norm(lang_pred, dim=1) * torch.linalg.norm(gt_lang, dim=1)
        )
        cos_dist = 1 - cos_sim
        return cos_dist.mean()

    def contrastive_lang_loss(self, seq_vis_feat, encoded_lang, use_for_aux_loss):
        assert self.lang_discriminator is not None
        if use_for_aux_loss is not None:
            if not torch.any(use_for_aux_loss):
                return torch.tensor(0.0).to(self.device)
            seq_vis_feat = seq_vis_feat[use_for_aux_loss]
            encoded_lang = encoded_lang[use_for_aux_loss]
        image_features, lang_features = self.proj_vis_lang(seq_vis_feat, encoded_lang)
        # l2 normalize embeddings?

        pred_pos = self.lang_discriminator(image_features, lang_features)
        labels_pos = torch.ones(pred_pos.shape, dtype=torch.float32, device=encoded_lang.device)
        labels_neg = torch.zeros(pred_pos.shape, dtype=torch.float32, device=encoded_lang.device)
        shifted_lang = torch.roll(lang_features, shifts=1, dims=0)
        pred_neg = self.lang_discriminator(image_features, shifted_lang)
        labels = torch.cat([labels_pos, labels_neg], 0)
        pred = torch.cat([pred_pos, pred_neg], 0)
        bce_loss = binary_cross_entropy_with_logits(pred, labels)
        return bce_loss

    def clip_loss(self, seq_vis_feat, encoded_lang, use_for_aux_loss):
        assert self.img_lang_matching_clip is not None
        if use_for_aux_loss is not None:
            if not torch.any(use_for_aux_loss):
                return torch.tensor(0.0).to(self.device)
            seq_vis_feat = seq_vis_feat[use_for_aux_loss]
            encoded_lang = encoded_lang[use_for_aux_loss]
        image_features, lang_features = self.proj_vis_lang(seq_vis_feat, encoded_lang)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = lang_features / lang_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # symmetric loss function
        labels = torch.arange(logits_per_image.shape[0], device=text_features.device)
        loss_i = cross_entropy(logits_per_image, labels)
        loss_t = cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2
        return loss

    def on_fit_start(self) -> None:
        if self.img_lang_matching_clip:
            train_dataset = self.trainer.datamodule.train_datasets["lang"]  # type: ignore
            val_dataset = self.trainer.datamodule.val_datasets["lang"]  # type: ignore
            self.val_dataset = val_dataset
            lang_data_train = np.load(
                train_dataset.abs_datasets_dir / train_dataset.lang_folder / "auto_lang_ann.npy", allow_pickle=True
            ).item()
            self.lang_data_val = np.load(
                val_dataset.abs_datasets_dir / val_dataset.lang_folder / "auto_lang_ann.npy", allow_pickle=True
            ).item()
            lang_embeddings_val = np.load(
                val_dataset.abs_datasets_dir / val_dataset.lang_folder / "embeddings.npy", allow_pickle=True
            ).item()
            train_lang_instructions = list(set(lang_data_train["language"]["ann"]))
            train_lang_ids = [
                lang_data_train["language"]["ann"].index(instruction) for instruction in train_lang_instructions
            ]
            self.train_lang_emb = (
                torch.from_numpy(lang_data_train["language"]["emb"][train_lang_ids]).to(self.device).squeeze().float()
            )
            train_lang_tasks = list(np.array(lang_data_train["language"]["task"])[train_lang_ids])
            train_lang_task_ids = [list(set(train_lang_tasks)).index(task) for task in train_lang_tasks]

            self.task_to_id = {k: v for k, v in zip(set(train_lang_tasks), set(train_lang_task_ids))}
            self.train_lang_task_ids = np.array(train_lang_task_ids)
            val_lang_tasks = []
            val_lang_emb = []
            val_lang_instructions = []
            for val_task, val_instructions in self.val_instructions.items():
                if val_task not in self.task_to_id:
                    continue
                val_lang_tasks.append(val_task)
                val_lang_emb.append(torch.from_numpy(lang_embeddings_val[val_task]["emb"][0]).to(self.device))
                val_lang_instructions.append(list(lang_embeddings_val[val_task]["ann"])[0])
            self.val_lang_emb = torch.cat(val_lang_emb).float()
            self.val_lang_task_ids = np.array([self.task_to_id[task] for task in val_lang_tasks])

    def validation_step(  # type: ignore
        self,
        batch: Dict[
            str,
            Dict,
        ],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        batch: list( batch_dataset_vision, batch_dataset_lang, ..., batch_dataset_differentModalities)
            - batch_dataset_vision: tuple( train_obs: Tensor,
                                           train_rgbs: tuple(Tensor, ),
                                           train_depths: tuple(Tensor, ),
                                           train_acts: Tensor ),
                                           info: Dict,
                                           idx: int
            - batch_dataset_lang: tuple( train_obs: Tensor,
                                         train_rgbs: tuple(Tensor, ),
                                         train_depths: tuple(Tensor, ),
                                         train_acts: Tensor,
                                         train_lang: Tensor   ),
                                         info: Dict,
                                         idx: int
        """
        output = {}
        val_total_act_loss_pp = torch.tensor(0.0).to(self.device)
        for self.modality_scope, dataset_batch in batch.items():
            perceptual_emb = self.perceptual_encoder(
                dataset_batch["rgb_obs"], dataset_batch["depth_obs"], dataset_batch["robot_obs"]
            )
            if self.state_recons:
                state_recon_loss = self.perceptual_encoder.state_reconstruction_loss()
                self.log(f"val/proprio_loss_{self.modality_scope}", state_recon_loss, sync_dist=True)
            if "lang" in self.modality_scope:
                latent_goal = self.language_goal(dataset_batch["lang"])
            else:
                latent_goal = self.visual_goal(perceptual_emb[:, -1])

            (
                sampled_plan_pp,
                action_loss_pp,
                sampled_plan_pr,
                action_loss_pr,
                kl_loss,
                mae_pp,
                mae_pr,
                gripper_sr_pp,
                gripper_sr_pr,
                seq_feat,
            ) = self.lmp_val(
                perceptual_emb, latent_goal, dataset_batch["actions"], dataset_batch["state_info"]["robot_obs"]
            )
            if "lang" in self.modality_scope:
                if self.lang_recons:
                    val_pred_lang_loss = self.lang_regression_loss(
                        seq_feat, dataset_batch["lang"], dataset_batch["use_for_aux_lang_loss"]
                    )
                    self.log("val/lang_pred_loss", val_pred_lang_loss, sync_dist=True)
                if self.img_lang_matching_clip:
                    val_pred_clip_loss = self.clip_loss(seq_feat, latent_goal, dataset_batch["use_for_aux_lang_loss"])
                    self.log("val/val_pred_clip_loss", val_pred_clip_loss, sync_dist=True)
                    self.clip_groundtruth(seq_feat, dataset_batch["idx"], dataset_batch["use_for_aux_lang_loss"])
                if self.lang_contrastive:
                    val_pred_contrastive_loss = self.contrastive_lang_loss(
                        seq_feat, latent_goal, dataset_batch["use_for_aux_lang_loss"]
                    )
                    self.log("val/lang_contrastive_loss", val_pred_contrastive_loss, sync_dist=True)
            val_total_act_loss_pp += action_loss_pp
            pr_mae_mean = mae_pr.mean()
            pp_mae_mean = mae_pp.mean()
            pos_mae_pp = mae_pp[..., :3].mean()
            pos_mae_pr = mae_pr[..., :3].mean()
            orn_mae_pp = mae_pp[..., 3:6].mean()
            orn_mae_pr = mae_pr[..., 3:6].mean()
            self.log(f"val_total_mae/{self.modality_scope}_total_mae_pr", pr_mae_mean, sync_dist=True)
            self.log(f"val_total_mae/{self.modality_scope}_total_mae_pp", pp_mae_mean, sync_dist=True)
            self.log(f"val_pos_mae/{self.modality_scope}_pos_mae_pr", pos_mae_pr, sync_dist=True)
            self.log(f"val_pos_mae/{self.modality_scope}_pos_mae_pp", pos_mae_pp, sync_dist=True)
            self.log(f"val_orn_mae/{self.modality_scope}_orn_mae_pr", orn_mae_pr, sync_dist=True)
            self.log(f"val_orn_mae/{self.modality_scope}_orn_mae_pp", orn_mae_pp, sync_dist=True)
            self.log(f"val_kl/{self.modality_scope}_kl_loss", kl_loss, sync_dist=True)
            self.log(f"val_act/{self.modality_scope}_act_loss_pp", action_loss_pp, sync_dist=True)
            self.log(f"val_act/{self.modality_scope}_act_loss_pr", action_loss_pr, sync_dist=True)
            self.log(f"val_grip/{self.modality_scope}_grip_sr_pr", gripper_sr_pr, sync_dist=True)
            self.log(f"val_grip/{self.modality_scope}_grip_sr_pp", gripper_sr_pp, sync_dist=True)
            self.log(
                "val_act/action_loss_pp",
                val_total_act_loss_pp / len(self.trainer.datamodule.modalities),  # type:ignore
                sync_dist=True,
            )
            output[f"sampled_plan_pp_{self.modality_scope}"] = sampled_plan_pp
            output[f"sampled_plan_pr_{self.modality_scope}"] = sampled_plan_pr
            output[f"idx_{self.modality_scope}"] = dataset_batch["idx"]

        return output

    def reset(self):
        self.plan = None
        self.latent_goal = None
        self.rollout_step_counter = 0

    def step(self, obs, goal):
        # replan every replan_freq steps (default 30 i.e every second)
        if self.rollout_step_counter % self.replan_freq == 0:
            if "lang" in goal:
                self.plan, self.latent_goal = self.get_pp_plan_lang(obs, goal)
            else:
                self.plan, self.latent_goal = self.get_pp_plan_vision(obs, goal)
        # use plan to predict actions with current observations
        action = self.predict_with_plan(obs, self.latent_goal, self.plan)
        self.rollout_step_counter += 1
        return action

    def predict_with_plan(
        self,
        obs: Dict[str, Any],
        latent_goal: torch.Tensor,
        sampled_plan: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            perceptual_emb = self.perceptual_encoder(obs["rgb_obs"], obs["depth_obs"], obs["robot_obs"])
            action = self.action_decoder.act(
                sampled_plan, perceptual_emb, latent_goal, obs["robot_obs_raw"]
            )  # type:  ignore

        return action

    def get_pp_plan_vision(self, obs: dict, goal: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(obs["rgb_obs"]) == len(goal["rgb_obs"])
        assert len(obs["depth_obs"]) == len(goal["depth_obs"])
        imgs = {k: torch.cat([v, goal["rgb_obs"][k]], dim=1) for k, v in obs["rgb_obs"].items()}  # (1, 2, C, H, W)
        depth_imgs = {k: torch.cat([v, goal["depth_obs"][k]], dim=1) for k, v in obs["depth_obs"].items()}
        state = torch.cat([obs["robot_obs"], goal["robot_obs"]], dim=1)
        with torch.no_grad():
            perceptual_emb = self.perceptual_encoder(imgs, depth_imgs, state)
            latent_goal = self.visual_goal(perceptual_emb[:, -1])
            # ------------Plan Proposal------------ #
            pp_state = self.plan_proposal(perceptual_emb[:, 0], latent_goal)
            pp_dist = self.dist.get_dist(pp_state)
            sampled_plan = self.dist.sample_latent_plan(pp_dist)
        self.action_decoder.clear_hidden_state()
        return sampled_plan, latent_goal

    def get_pp_plan_lang(self, obs: dict, goal: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            perceptual_emb = self.perceptual_encoder(obs["rgb_obs"], obs["depth_obs"], obs["robot_obs"])
            latent_goal = self.language_goal(goal["lang"])
            # ------------Plan Proposal------------ #
            pp_state = self.plan_proposal(perceptual_emb[:, 0], latent_goal)
            pp_dist = self.dist.get_dist(pp_state)
            sampled_plan = self.dist.sample_latent_plan(pp_dist)
        self.action_decoder.clear_hidden_state()
        return sampled_plan, latent_goal

    @rank_zero_only
    def on_train_epoch_start(self) -> None:
        logger.info(f"Start training epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_end(self, unused: Optional = None) -> None:  # type: ignore
        logger.info(f"Finished training epoch {self.current_epoch}")

    def on_validation_epoch_start(self) -> None:
        log_rank_0(f"Start validation epoch {self.current_epoch}")
        if self.img_lang_matching_clip:
            if self.train_lang_emb.device != self.device:  # type: ignore
                self.train_lang_emb = self.train_lang_emb.to(self.device)  # type: ignore
                self.val_lang_emb = self.val_lang_emb.to(self.device)  # type: ignore
            self.encoded_lang_train = self.language_goal(self.train_lang_emb)
            self.encoded_lang_val = self.language_goal(self.val_lang_emb)

    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        logger.info(f"Finished validation epoch {self.current_epoch}")

    def clip_inference(self, obs: dict, goal: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            perceptual_emb = self.perceptual_encoder(obs["rgb_obs"], obs["depth_obs"], obs["robot_obs"])
            encoded_lang = self.language_goal(goal["lang"])
            _, seq_vis_feat = self.plan_recognition(perceptual_emb)
            image_features, lang_features = self.proj_vis_lang(seq_vis_feat, encoded_lang)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = lang_features / lang_features.norm(dim=-1, keepdim=True)
            # cosine similarity as logits
            logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()
            return logits_per_image, logits_per_text

    def clip_groundtruth(self, seq_feat_vis, idx, use_for_aux_loss):
        if use_for_aux_loss is not None and not torch.any(use_for_aux_loss):
            return
        seq_feat_vis = seq_feat_vis[use_for_aux_loss]
        gt_tasks = [
            self.task_to_id[self.lang_data_val["language"]["task"][self.val_dataset.lang_lookup[i]]] for i in idx
        ]
        gt_tasks = np.array(gt_tasks)[use_for_aux_loss.cpu().numpy()]

        train_score, train_sr = self._clip_groundtruth_loss(
            seq_feat_vis, self.encoded_lang_train, self.train_lang_task_ids, gt_tasks
        )
        val_score, val_sr = self._clip_groundtruth_loss(
            seq_feat_vis, self.encoded_lang_val, self.val_lang_task_ids, gt_tasks
        )
        self.log("lang_gt/train_gt", train_score, sync_dist=True)
        self.log("lang_gt/val_gt", val_score, sync_dist=True)
        self.log("lang_gt/train_sr", train_sr, sync_dist=True)
        self.log("lang_gt/val_sr", val_sr, sync_dist=True)

    def _clip_groundtruth_loss(self, seq_feat_vis, encoded_lang, task_ids, gt_tasks):

        image_features, lang_features = self.proj_vis_lang(seq_feat_vis, encoded_lang)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = lang_features / lang_features.norm(dim=-1, keepdim=True)
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        scores = logits_per_image
        scores -= torch.min(scores, dim=1)[0].unsqueeze(1)
        scores /= torch.max(scores, dim=1)[0].unsqueeze(1) - torch.min(scores, dim=1)[0].unsqueeze(1)

        loss = []

        for score, gt_task in zip(scores, gt_tasks):
            positive_ids = np.where(task_ids == gt_task)[0]
            negative_ids = np.where(task_ids != gt_task)[0]
            loss.append(torch.sum(score[positive_ids]) - torch.sum(score[negative_ids]))

        loss = torch.mean(torch.stack(loss))

        sr = np.mean(task_ids[torch.argmax(scores, dim=1).cpu()] == gt_tasks)
        return loss, sr
