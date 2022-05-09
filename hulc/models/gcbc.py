import logging
from typing import Dict

import torch

from hulc.models.hulc import Hulc

logger = logging.getLogger(__name__)


class GCBC(Hulc):
    """
    Goal-conditioned behavior cloning.
    """

    @staticmethod
    def setup_input_sizes(
        perceptual_encoder,
        plan_proposal,
        plan_recognition,
        visual_goal,
        action_decoder,
        distribution,
    ):
        """
        Configure the input feature sizes of the respective parts of the network.

        Args:
            perceptual_encoder: DictConfig for perceptual encoder.
            plan_proposal: DictConfig for plan proposal network.
            plan_recognition: DictConfig for plan recognition network.
            visual_goal: DictConfig for visual goal encoder.
            action_decoder: DictConfig for action decoder network.
            distribution: DictConfig for plan distribution (continuous or discrete).
        """
        plan_proposal.perceptual_features = perceptual_encoder.latent_size
        plan_recognition.in_features = perceptual_encoder.latent_size
        visual_goal.in_features = perceptual_encoder.latent_size
        action_decoder.perceptual_features = perceptual_encoder.latent_size

        if distribution.dist == "discrete":
            plan_proposal.plan_features = distribution.class_size * distribution.category_size
            plan_recognition.plan_features = distribution.class_size * distribution.category_size
            action_decoder.plan_features = 0
        elif distribution.dist == "continuous":
            plan_proposal.plan_features = distribution.plan_features
            plan_recognition.plan_features = distribution.plan_features
            action_decoder.plan_features = 0

    def training_step(self, batch: Dict[str, Dict], batch_idx: int) -> torch.Tensor:  # type: ignore
        """
        Compute and return the training loss.

        Args:
            batch (dict):
                - 'vis' (dict):
                    - 'rgb_obs' (dict):
                        - 'rgb_static' (Tensor): RGB camera image of static camera
                        - ...
                    - 'depth_obs' (dict):
                        - 'depth_static' (Tensor): Depth camera image of depth camera
                        - ...
                    - 'robot_obs' (Tensor): Proprioceptive state observation.
                    - 'actions' (Tensor): Ground truth actions.
                    - 'state_info' (dict):
                        - 'robot_obs' (Tensor): Unnormalized robot states.
                        - 'scene_obs' (Tensor): Unnormalized scene states.
                    - 'idx' (LongTensor): Episode indices.
                - 'lang' (dict):
                    Like 'vis' but with additional keys:
                        - 'language' (Tensor): Embedded Language labels.
                        - 'use_for_aux_lang_loss' (BoolTensor): Mask of which sequences in the batch to consider for
                            auxiliary loss.
            batch_idx (int): Integer displaying index of this batch.


        Returns:
            loss tensor
        """
        action_loss, proprio_loss, lang_pred_loss, lang_contrastive_loss, lang_clip_loss, total_loss = (
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
        )

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

            robot_obs = dataset_batch["state_info"]["robot_obs"]
            actions = dataset_batch["actions"]
            empty_plan = torch.empty((dataset_batch["actions"].shape[0]), 0).to(self.device)
            act_loss = self.action_decoder.loss(empty_plan, perceptual_emb, latent_goal, actions, robot_obs)
            _, seq_feat = self.plan_recognition(perceptual_emb)

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
            action_loss += act_loss
            total_loss += act_loss
            batch_size[self.modality_scope] = dataset_batch["actions"].shape[0]
            total_bs += dataset_batch["actions"].shape[0]

            self.log(
                f"train/action_loss_{self.modality_scope}",
                act_loss,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size[self.modality_scope],
            )
        total_loss = total_loss / len(batch)  # divide accumulated gradients by number of datasets
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
        self.log("train/action_loss", action_loss, on_step=False, on_epoch=True, batch_size=total_bs)
        self.log("train/total_loss", total_loss, on_step=False, on_epoch=True, batch_size=total_bs)
        return total_loss

    def validation_step(self, batch: Dict[str, Dict], batch_idx: int) -> Dict[str, torch.Tensor]:  # type: ignore
        """
        Compute and log the validation losses and additional metrics.

        Args:
            batch (dict):
                - 'vis' (dict):
                    - 'rgb_obs' (dict):
                        - 'rgb_static' (Tensor): RGB camera image of static camera
                        - ...
                    - 'depth_obs' (dict):
                        - 'depth_static' (Tensor): Depth camera image of depth camera
                        - ...
                    - 'robot_obs' (Tensor): Proprioceptive state observation.
                    - 'actions' (Tensor): Ground truth actions.
                    - 'state_info' (dict):
                        - 'robot_obs' (Tensor): Unnormalized robot states.
                        - 'scene_obs' (Tensor): Unnormalized scene states.
                    - 'idx' (LongTensor): Episode indices.
                - 'lang' (dict):
                    Like 'vis' but with additional keys:
                        - 'language' (Tensor): Embedded Language labels.
                        - 'use_for_aux_lang_loss' (BoolTensor): Mask of which sequences in the batch to consider for
                            auxiliary loss.
            batch_idx (int): Integer displaying index of this batch.

        Returns:
            Dictionary containing the sampled plans of plan recognition and plan proposal networks, as well as the
            episode indices.
        """
        output = {}
        val_total_act_loss = torch.tensor(0.0).to(self.device)
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

            robot_obs = dataset_batch["state_info"]["robot_obs"]
            actions = dataset_batch["actions"]
            empty_plan = torch.empty((dataset_batch["actions"].shape[0]), 0).to(self.device)
            action_loss, sample_act = self.action_decoder.loss_and_act(  # type:  ignore
                empty_plan, perceptual_emb, latent_goal, actions, robot_obs
            )
            mae = torch.nn.functional.l1_loss(
                sample_act[..., :-1], actions[..., :-1], reduction="none"
            )  # (batch, seq, 6)
            mae = torch.mean(mae, 1)  # (batch, 6)
            # gripper action
            gripper_discrete = sample_act[..., -1]
            gt_gripper_act = actions[..., -1]
            m = gripper_discrete > 0
            gripper_discrete[m] = 1
            gripper_discrete[~m] = -1
            gripper_sr = torch.mean((gt_gripper_act == gripper_discrete).float())
            _, seq_feat = self.plan_recognition(perceptual_emb)

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
            val_total_act_loss += action_loss
            mae_mean = mae.mean()
            pos_mae = mae[..., :3].mean()
            orn_mae = mae[..., 3:6].mean()
            self.log(f"val_total_mae/{self.modality_scope}_total_mae", mae_mean, sync_dist=True)
            self.log(f"val_pos_mae/{self.modality_scope}_pos_mae", pos_mae, sync_dist=True)
            self.log(f"val_orn_mae/{self.modality_scope}_orn_mae", orn_mae, sync_dist=True)
            self.log(f"val_act/{self.modality_scope}_act_loss", action_loss, sync_dist=True)
            self.log(f"val_grip/{self.modality_scope}_grip_sr", gripper_sr, sync_dist=True)
            self.log(
                "val_act/action_loss",
                val_total_act_loss / len(self.trainer.datamodule.modalities),  # type:ignore
                sync_dist=True,
            )
            output[f"idx_{self.modality_scope}"] = dataset_batch["idx"]

        return output

    def reset(self):
        """
        Call this at the beginning of a new rollout when doing inference.
        """
        self.latent_goal = None

    def step(self, obs, goal):
        """
        Do one step of inference with the model.

        Args:
            obs (dict): Observation from environment.
            goal (dict): Goal as visual observation or embedded language instruction.

        Returns:
            Predicted action.
        """
        with torch.no_grad():
            if self.latent_goal is None:
                if "lang" in goal:
                    self.latent_goal = self.language_goal(goal["lang"])
                else:
                    imgs = {
                        k: torch.cat([v, goal["rgb_obs"][k]], dim=1) for k, v in obs["rgb_obs"].items()
                    }  # (1, 2, C, H, W)
                    depth_imgs = {k: torch.cat([v, goal["depth_obs"][k]], dim=1) for k, v in obs["depth_obs"].items()}
                    state = torch.cat([obs["robot_obs"], goal["robot_obs"]], dim=1)
                    perceptual_emb = self.perceptual_encoder(imgs, depth_imgs, state)
                    self.latent_goal = self.visual_goal(perceptual_emb[:, -1])

            perceptual_emb = self.perceptual_encoder(obs["rgb_obs"], obs["depth_obs"], obs["robot_obs"])
            empty_plan = torch.empty(1, 0).to(self.device)
            action = self.action_decoder.act(
                empty_plan, perceptual_emb, self.latent_goal, obs["robot_obs_raw"]
            )  # type:  ignore
            return action
