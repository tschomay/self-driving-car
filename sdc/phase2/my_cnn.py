"""The owner's CNN encoder (learner-owned code, Phase 2 E5 and gate).

It turns the camera image(s) plus the speed number into one feature vector. The policy
head (a small MLP, SB3's) turns that vector into steering and throttle. It is trained by
behavior cloning (sdc/phase2/bc.py --encoder mine), the recipe that worked in E3.

How a convolution layer works (one line per idea):
  - A *filter* is a small grid of weights (the kernel, e.g. 5×5) that slides over the image.
  - At each position it multiplies the pixels under it by its weights and sums them: one number.
    Where the patch looks like the pattern in the weights (say, a bright line at 45°), the number is big.
  - Sliding it over the whole image gives a *feature map*: where that pattern appears.
  - A layer has several filters (its *channels*), so it outputs several maps, one per pattern.
  - *Stride* is how far the filter jumps between positions. Stride 2 halves the map's width and height.
  - The next layer's filters look at the previous layer's maps, so they find patterns of patterns
    (edge pieces → a curving edge → "the road bends left ahead").
  - The *receptive field* is how much of the original image one output number can see. It grows with
    each layer, faster with bigger kernels and strides.
  - After each layer comes a simple nonlinearity (e.g. ReLU: negative → 0), without which stacked layers
    would collapse into one.
  - At the end the maps are flattened into one long vector and squeezed by a linear layer to
    `features_dim` numbers.

Map size after a layer (no padding): out = (in - kernel) // stride + 1.

OWNER TODOs below. The agent wrote the plumbing (sizes, speed input, wiring into SB3).
"""

from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


def conv_layers(in_channels: int) -> nn.Sequential | None:
    """OWNER TODO 1: the convolution stack.

    Input: (batch, in_channels, 64, 64), pixel values scaled to 0..1 (in_channels = stacked frames).
    Return an nn.Sequential of Conv2d layers, each followed by the activation (OWNER TODO 2).
    Example of the syntax only: nn.Conv2d(in_channels, 16, kernel_size=5, stride=2), nn.ReLU()

    Owner's design (2026-11-09), worked out with the agent: fine detail first, then shrink.
      layer  filters  kernel  stride   map     each output sees
      1      32       5×5     2        30×30   5 px  (3 m)
      2      64       3×3     2        14×14   9 px  (6 m)
      3      64       3×3     2        6×6     17 px (11 m)
      4      64       3×3     1        4×4     33 px (21 m)  → 1,024 numbers into the squeeze
    (A first version with stride 1 in layers 2-4 gave 26×26 maps, 43k numbers, 11M weights and
    8 m per output; the owner wanted features that see further at an affordable cost.)
    """
    act = activation
    if act() is None:
        return None
    return nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=5, stride=2), act(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2), act(),
        nn.Conv2d(64, 64, kernel_size=3, stride=2), act(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1), act(),
    )


def activation() -> nn.Module | None:
    """OWNER TODO 2: the nonlinearity after each layer (e.g. nn.ReLU(), nn.Tanh(), nn.ELU())."""
    return None


FEATURES_DIM = None   # OWNER TODO 3: size of the vector the image is squeezed into (e.g. 64, 128, 256)


class MyEncoder(BaseFeaturesExtractor):
    """Camera image(s) → conv_layers → flatten → linear (FEATURES_DIM) → activation, then append speed."""

    def __init__(self, observation_space: spaces.Dict):
        img_space = observation_space["image"]
        self.use_speed = "speed" in observation_space.spaces
        convs = conv_layers(img_space.shape[0])
        if convs is None or FEATURES_DIM is None:
            raise NotImplementedError("sdc/phase2/my_cnn.py: OWNER TODOs 1-3 not filled in yet")
        super().__init__(observation_space, FEATURES_DIM + int(self.use_speed))
        with torch.no_grad():
            n_flat = convs(torch.zeros(1, *img_space.shape)).flatten(1).shape[1]
        self.convs = convs
        self.linear = nn.Sequential(nn.Linear(n_flat, FEATURES_DIM), activation())
        self.n_flat = n_flat

    def forward(self, obs: dict) -> torch.Tensor:
        x = self.linear(self.convs(obs["image"]).flatten(1))   # SB3 already scaled the image to 0..1
        return torch.cat([x, obs["speed"]], dim=1) if self.use_speed else x

    def maps(self, image: torch.Tensor) -> list[torch.Tensor]:
        """Feature maps after each layer (for the saliency / what-it-looks-at report)."""
        out, x = [], image
        for layer in self.convs:
            x = layer(x)
            if not isinstance(layer, nn.Conv2d):
                out.append(x)
        return out
