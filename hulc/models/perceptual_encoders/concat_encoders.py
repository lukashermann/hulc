from typing import Dict, Optional

import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss


class ConcatEncoders(nn.Module):
    def __init__(
        self,
        rgb_static: DictConfig,
        proprio: DictConfig,
        device: torch.device,
        depth_static: Optional[DictConfig] = None,
        rgb_gripper: Optional[DictConfig] = None,
        depth_gripper: Optional[DictConfig] = None,
        tactile: Optional[DictConfig] = None,
        state_decoder: Optional[DictConfig] = None,
    ):
        super().__init__()
        self._latent_size = rgb_static.visual_features
        if rgb_gripper:
            self._latent_size += rgb_gripper.visual_features
        if depth_static:
            self._latent_size += depth_static.visual_features
        if depth_gripper:
            self._latent_size += depth_gripper.visual_features
        if tactile:
            self._latent_size += tactile.visual_features
        visual_features = self._latent_size
        # super ugly, fix this clip ddp thing in a better way
        if "clip" in rgb_static["_target_"]:
            self.rgb_static_encoder = hydra.utils.instantiate(rgb_static, device=device)
        else:
            self.rgb_static_encoder = hydra.utils.instantiate(rgb_static)
        self.depth_static_encoder = hydra.utils.instantiate(depth_static) if depth_static else None
        self.rgb_gripper_encoder = hydra.utils.instantiate(rgb_gripper) if rgb_gripper else None
        self.depth_gripper_encoder = hydra.utils.instantiate(depth_gripper) if depth_gripper else None
        self.tactile_encoder = hydra.utils.instantiate(tactile)
        self.proprio_encoder = hydra.utils.instantiate(proprio)
        if self.proprio_encoder:
            self._latent_size += self.proprio_encoder.out_features

        self.state_decoder = None
        if state_decoder:
            state_decoder.visual_features = visual_features
            state_decoder.n_state_obs = self.proprio_encoder.out_features
            self.state_decoder = hydra.utils.instantiate(state_decoder)

        self.current_visual_embedding = None
        self.current_state_obs = None

    @property
    def latent_size(self):
        return self._latent_size

    def forward(
        self, imgs: Dict[str, torch.Tensor], depth_imgs: Dict[str, torch.Tensor], state_obs: torch.Tensor
    ) -> torch.Tensor:
        rgb_static = imgs["rgb_static"]
        rgb_gripper = imgs["rgb_gripper"] if "rgb_gripper" in imgs else None
        rgb_tactile = imgs["rgb_tactile"] if "rgb_tactile" in imgs else None
        depth_static = depth_imgs["depth_static"] if "depth_static" in depth_imgs else None
        depth_gripper = depth_imgs["depth_gripper"] if "depth_gripper" in depth_imgs else None

        b, s, c, h, w = rgb_static.shape
        rgb_static = rgb_static.reshape(-1, c, h, w)  # (batch_size * sequence_length, 3, 200, 200)
        # ------------ Vision Network ------------ #
        encoded_imgs = self.rgb_static_encoder(rgb_static)  # (batch*seq_len, 64)
        encoded_imgs = encoded_imgs.reshape(b, s, -1)  # (batch, seq, 64)

        if depth_static is not None:
            depth_static = torch.unsqueeze(depth_static, 2)
            depth_static = depth_static.reshape(-1, 1, h, w)  # (batch_size * sequence_length, 3, 200, 200)
            encoded_depth_static = self.depth_static_encoder(depth_static)  # (batch*seq_len, 64)
            encoded_depth_static = encoded_depth_static.reshape(b, s, -1)  # (batch, seq, 64)
            encoded_imgs = torch.cat([encoded_imgs, encoded_depth_static], dim=-1)

        if rgb_gripper is not None:
            b, s, c, h, w = rgb_gripper.shape
            rgb_gripper = rgb_gripper.reshape(-1, c, h, w)  # (batch_size * sequence_length, 3, 84, 84)
            encoded_imgs_gripper = self.rgb_gripper_encoder(rgb_gripper)  # (batch*seq_len, 64)
            encoded_imgs_gripper = encoded_imgs_gripper.reshape(b, s, -1)  # (batch, seq, 64)
            encoded_imgs = torch.cat([encoded_imgs, encoded_imgs_gripper], dim=-1)
            if depth_gripper is not None:
                depth_gripper = torch.unsqueeze(depth_gripper, 2)
                depth_gripper = depth_gripper.reshape(-1, 1, h, w)  # (batch_size * sequence_length, 1, 84, 84)
                encoded_depth_gripper = self.depth_gripper_encoder(depth_gripper)
                encoded_depth_gripper = encoded_depth_gripper.reshape(b, s, -1)  # (batch, seq, 64)
                encoded_imgs = torch.cat([encoded_imgs, encoded_depth_gripper], dim=-1)

        if rgb_tactile is not None:
            b, s, c, h, w = rgb_tactile.shape
            rgb_tactile = rgb_tactile.reshape(-1, c, h, w)  # (batch_size * sequence_length, 3, 84, 84)
            encoded_tactile = self.tactile_encoder(rgb_tactile)
            encoded_tactile = encoded_tactile.reshape(b, s, -1)
            encoded_imgs = torch.cat([encoded_imgs, encoded_tactile], dim=-1)

        self.current_visual_embedding = encoded_imgs
        self.current_state_obs = state_obs  # type: ignore
        if self.proprio_encoder:
            state_obs_out = self.proprio_encoder(state_obs)
            perceptual_emb = torch.cat([encoded_imgs, state_obs_out], dim=-1)
        else:
            perceptual_emb = encoded_imgs

        return perceptual_emb

    def state_reconstruction_loss(self):
        assert self.state_decoder is not None
        proprio_pred = self.state_decoder(self.current_visual_embedding)
        return mse_loss(self.current_state_obs, proprio_pred)
