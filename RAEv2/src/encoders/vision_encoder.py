from typing import Dict, List, Optional

import torch
import torch.nn as nn



class VisionEncoder(nn.Module):
    """Base class for all vision encoders"""

    def __init__(self, encoder_type: str, architecture: str, model_config: str,
                 device: torch.device, resolution: int = 256, accelerator=None):
        super().__init__()  # Initialize nn.Module
        self.encoder_type = encoder_type
        self.architecture = architecture
        self.model_config = model_config
        self.device = device
        self.resolution = resolution
        self.accelerator = accelerator
        self._embed_dim = None
        self.model = None
        self.patch_size = None  # Subclasses should set this

    def load_model(self):
        """Load and initialize the encoder model - subclasses should override"""
        raise NotImplementedError("Subclasses must implement load_model()")

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess raw images - subclasses should override
        Args:
            x: Raw images tensor (B, C, H, W) in range [0, 255]
        Returns:
            Preprocessed tensor ready for encoder
        """
        raise NotImplementedError("Subclasses must implement preprocess()")

    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        """
        Forward pass through encoder
        Args:
            x: Preprocessed images
        Returns:
            Dictionary with:
                - 'x_norm_clstoken': (B, D) CLS token or None if not available
                - 'x_norm_patchtokens': (B, T, D) patch tokens
        """
        # Default implementation - subclasses should override if needed
        out = self.model.forward_features(x)
        if isinstance(out, dict):
            return out
        else:
            # Assume it's just patch tokens
            return {
                'x_norm_clstoken': None,
                'x_norm_patchtokens': out
            }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        RAE-compatible forward pass returning only patch tokens.

        Args:
            x: Input images (B, C, H, W)

        Returns:
            Patch tokens (B, T, D)
        """
        x = self.preprocess(x)
        features = self.forward_features(x)
        return features['x_norm_patchtokens']

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def hidden_size(self) -> int:
        return self._embed_dim

    def eval(self):
        """Set model to eval mode"""
        if self.model is not None:
            self.model.eval()
        return self

    def to(self, device):
        """Move model to device"""
        if self.model is not None:
            self.model = self.model.to(device)
        self.device = device
        return self


class DINOv3Encoder(VisionEncoder):
    """DINOv3 encoder implementation.

    Supports optional flags in model_config: e.g., 'b16[norm]'
      - Default (no flags): norm_affine=False (matches DINOv2 default)
      - [norm]: keep layernorm affine params
    """

    _KNOWN_FLAGS = {'norm'}
    _KNOWN_BASES = {'s16', 's16plus', 'b16', 'l16', 'h16plus', '7b16'}

    def _parse_config(self):
        """Parse model_config for base config and flags.

        DINOv3 base configs are multi-character (s16, b16, l16, etc.).
        Supports:
            'b16'           -> base='b16', flags=set()
            'b16[norm]'     -> base='b16', flags={'norm'}
            'b16norm'       -> base='b16', flags={'norm'}
        """
        import re
        # Bracket syntax: e.g. 'b16[norm]'
        match = re.match(r'^(.+?)\[([^\]]+)\]$', self.model_config)
        if match:
            base = match.group(1)
            flags = set(f.strip() for f in match.group(2).split(','))
            return base, flags

        # Concatenated suffix: match longest known base, parse flags from remainder
        cfg = self.model_config
        best_base = None
        for known_base in sorted(self._KNOWN_BASES, key=len, reverse=True):
            if cfg.startswith(known_base):
                best_base = known_base
                break

        if best_base:
            suffix = cfg[len(best_base):]
            if not suffix:
                return best_base, set()
            flags = set()
            remaining = suffix
            while remaining:
                matched = False
                for flag in self._KNOWN_FLAGS:
                    if remaining.startswith(flag):
                        flags.add(flag)
                        remaining = remaining[len(flag):]
                        matched = True
                        break
                if not matched:
                    return self.model_config, set()
            return best_base, flags

        return self.model_config, set()

    def load_model(self):
        from .dinov3_loader import load_dinov3

        base_config, flags = self._parse_config()
        use_norm_affine = 'norm' in flags

        self.model = load_dinov3(f"dinov3_vit{base_config}")
        self.model = self.model.to(self.device)
        self.model.eval()

        # Set embed dim and patch size
        self._embed_dim = self.model.embed_dim
        self.patch_size = 16

        # Strip norm affine by default (matches DINOv2 default)
        if not use_norm_affine:
            self.model.norm = nn.LayerNorm(self._embed_dim, elementwise_affine=False)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        from .dinov3_loader import make_dinov3_transform
        transform_func = make_dinov3_transform(resize_size=self.resolution)
        return transform_func(x)

    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        out = self.model.forward_features(x)
        return {
            'x_norm_clstoken': out.get('x_norm_clstoken'),
            'x_norm_patchtokens': out.get('x_norm_patchtokens')
        }


class DINOv3MultiLayerSimpleAddEncoder(DINOv3Encoder):
    """DINOv3 encoder that averages patch tokens from multiple layers.

    Same approach as DINOv2MultiLayerSimpleAddEncoder but for DINOv3 models.
    Config format: 'l16[layers=21.23]', 'b16[layers=7.9.11]'
    Default layers per model: l16=[5,11,17,23], b16=[2,5,8,11]
    """

    DEFAULT_LAYERS = {
        's16': [2, 5, 8, 11],
        'b16': [2, 5, 8, 11],
        'l16': [5, 11, 17, 23],
        'h16plus': [8, 16, 24, 31],
    }

    def load_model(self):
        super().load_model()
        base_config, flags = self._parse_config()
        layers_flag = [f for f in flags if f.startswith('layers=')]
        if layers_flag:
            self.layer_indices = [int(i) for i in layers_flag[0].split('=')[1].split('.')]
        else:
            self.layer_indices = self.DEFAULT_LAYERS.get(base_config, [2, 5, 8, 11])

    def _parse_config(self):
        import re
        match = re.match(r'^(.+?)\[([^\]]+)\]$', self.model_config)
        if match:
            base = match.group(1)
            flags = [f.strip() for f in match.group(2).split(',')]
            return base, flags
        return self.model_config, []

    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        outputs = self.model.get_intermediate_layers(
            x, n=self.layer_indices, reshape=False,
            return_class_token=False, norm=True
        )
        patch_tokens = torch.stack(outputs, dim=0).mean(dim=0)
        final_mean = outputs[-1].mean(dim=1, keepdim=True)
        patch_tokens = patch_tokens + final_mean
        return {
            'x_norm_clstoken': final_mean.squeeze(1),
            'x_norm_patchtokens': patch_tokens,
        }


ENCODER_REGISTRY = {
    "dinov3": DINOv3Encoder,
    "dinov3mls": DINOv3MultiLayerSimpleAddEncoder,
}


def create_encoder(encoder_string: str, device: torch.device, resolution: int = 256, accelerator=None) -> VisionEncoder:
    """encoder_string = "<type>-<architecture>-<model_config>", e.g. dinov3mls-vit-l16[layers=11.13.15.17.19.21.23]."""
    parts = encoder_string.split("-")
    if len(parts) != 3:
        raise ValueError(f"Invalid encoder string: {encoder_string}")
    encoder_type, architecture, model_config = parts
    if encoder_type not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder type {encoder_type}; available: {list(ENCODER_REGISTRY)}")
    encoder = ENCODER_REGISTRY[encoder_type](encoder_type, architecture, model_config, device, resolution, accelerator)
    encoder.load_model()
    return encoder
