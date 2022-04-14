import torch
from torch import nn
import torch.nn.functional as F

from hulc.models.perceptual_encoders.clip import load_clip


class VisionClip(nn.Module):
    def __init__(
        self, device: torch.device, visual_features: int, freeze_backbone: bool = True, model_name: str = "RN50"
    ):
        super(VisionClip, self).__init__()
        # Load CLIP model
        print(f"loading vision CLIP model with backbone: {model_name}")
        self.clip_model, _ = load_clip(model_name, device=device)
        if freeze_backbone:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        if "RN50" in model_name:
            self.fc1 = nn.Linear(1024, 512)
            self.fc2 = nn.Linear(512, visual_features)
        elif "ViT-B/32" in model_name:
            self.fc1 = nn.Linear(512, 256)
            self.fc2 = nn.Linear(256, visual_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.clip_model.encode_image(x)  # type:ignore
        output = F.relu(self.fc1(x))  # batch, 512
        output = self.fc2(output)  # batch, 64
        return output
