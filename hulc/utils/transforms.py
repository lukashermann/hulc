import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# source: https://github.com/facebookresearch/drqv2/blob/main/drqv2.py
class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class RelativeActions(object):
    """Transform absolute actions to relative"""

    def __init__(self, max_pos, max_orn):
        self.max_pos = max_pos
        self.max_orn = max_orn

    @staticmethod
    def batch_angle_between(a, b):
        diff = b - a
        return (diff + np.pi) % (2 * np.pi) - np.pi

    def __call__(self, action_and_obs):
        actions, robot_obs = action_and_obs
        assert isinstance(actions, np.ndarray)
        assert isinstance(robot_obs, np.ndarray)

        rel_pos = actions[:, :3] - robot_obs[:, :3]
        rel_pos = np.clip(rel_pos, -self.max_pos, self.max_pos) / self.max_pos

        rel_orn = self.batch_angle_between(robot_obs[:, 3:6], actions[:, 3:6])
        rel_orn = np.clip(rel_orn, -self.max_orn, self.max_orn) / self.max_orn

        gripper = actions[:, -1:]
        return np.concatenate([rel_pos, rel_orn, gripper], axis=1)

    def __repr__(self):
        return self.__class__.__name__ + f"(max_pos={self.max_pos}, max_orn={self.max_orn})"
