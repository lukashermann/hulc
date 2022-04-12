import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from omegaconf import ListConfig, OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F

import hulc
from hulc.models.decoders.action_decoder import ActionDecoder
from hulc.models.decoders.utils.gripper_control import tcp_to_world_frame, world_to_tcp_frame
from hulc.models.decoders.utils.rnn import gru_decoder, lstm_decoder, mlp_decoder, rnn_decoder

logger = logging.getLogger(__name__)


class DeterministicDecoder(ActionDecoder):
    def __init__(
        self,
        perceptual_features: int,
        latent_goal_features: int,
        plan_features: int,
        hidden_size: int,
        out_features: int,
        policy_rnn_dropout_p: float,
        criterion: str,
        num_layers: int,
        rnn_model: str,
        perceptual_emb_slice: tuple,
        gripper_control: bool,
    ):
        super(DeterministicDecoder, self).__init__()
        self.plan_features = plan_features
        self.gripper_control = gripper_control
        self.out_features = out_features
        in_features = (perceptual_emb_slice[1] - perceptual_emb_slice[0]) + latent_goal_features + plan_features
        self.rnn = eval(rnn_model)
        self.rnn = self.rnn(in_features, hidden_size, num_layers, policy_rnn_dropout_p)
        self.actions = nn.Sequential(nn.Linear(hidden_size, out_features), nn.Tanh())
        self.criterion = getattr(nn, criterion)()
        self.perceptual_emb_slice = perceptual_emb_slice
        self.hidden_state = None

    def clear_hidden_state(self) -> None:
        self.hidden_state = None

    def forward(  # type: ignore
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        h_0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        perceptual_emb = perceptual_emb[..., slice(*self.perceptual_emb_slice)]
        batch_size, seq_len = perceptual_emb.shape[0], perceptual_emb.shape[1]
        latent_plan = latent_plan.unsqueeze(1).expand(-1, seq_len, -1) if latent_plan.nelement() > 0 else latent_plan
        latent_goal = latent_goal.unsqueeze(1).expand(-1, seq_len, -1)
        x = torch.cat([latent_plan, perceptual_emb, latent_goal], dim=-1)  # b, s, (plan + visuo-propio + goal)
        if not isinstance(self.rnn, nn.Sequential) and isinstance(self.rnn, nn.RNNBase):
            x, h_n = self.rnn(x, h_0)
        else:
            x = self.rnn(x)
            h_n = None
        actions = self.actions(x)
        return actions, h_n

    def loss_and_act(
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        actions: torch.Tensor,
        robot_obs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_actions, _ = self(latent_plan, perceptual_emb, latent_goal)
        # loss
        if self.gripper_control:
            actions_tcp = world_to_tcp_frame(actions, robot_obs)
            loss = self.criterion(pred_actions, actions_tcp)
            pred_actions_world = tcp_to_world_frame(pred_actions, robot_obs)
            return loss, pred_actions_world
        else:
            loss = self.criterion(pred_actions, actions)
            return loss, pred_actions

    def loss(
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        actions: torch.Tensor,
        robot_obs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pred_actions, _ = self(latent_plan, perceptual_emb, latent_goal)
        if self.gripper_control:
            actions_tcp = world_to_tcp_frame(actions, robot_obs)
            self.criterion(pred_actions, actions_tcp)
        return self.criterion(pred_actions, actions)

    def act(
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        robot_obs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pred_actions, self.hidden_state = self(latent_plan, perceptual_emb, latent_goal, self.hidden_state)
        if self.gripper_control:
            pred_actions_world = tcp_to_world_frame(pred_actions, robot_obs)
            return pred_actions_world
        else:
            return pred_actions
