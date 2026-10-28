"""Qwen3.5 text stack as the predictor backbone.

The backbone consumes pre-assembled inputs_embeds (vision tokens, text tokens, query tokens) and
returns the last hidden states. Qwen3.5 is a hybrid causal decoder (3 of 4 layers gated DeltaNet
linear attention, 1 full attention); flash-linear-attention is required for training.
"""

import logging

import torch
import torch.nn as nn

from app.lala_world.models.hub import resolve_model_id

logger = logging.getLogger(__name__)


def _keep_first_layers(model, num_layers):
    layers, config = model.layers, model.config
    if not 1 <= num_layers <= len(layers):
        raise ValueError(f"num_layers must be in [1, {len(layers)}], got {num_layers}")
    del layers[num_layers:]
    config.num_hidden_layers = num_layers
    if getattr(config, "layer_types", None):
        config.layer_types = list(config.layer_types[:num_layers])
    logger.info(f"using the first {num_layers} LLM layers")


class Qwen35Backbone(nn.Module):
    def __init__(self, model_id, num_layers=None, pretrained=True):
        super().__init__()
        from transformers import AutoConfig, AutoTokenizer, Qwen3_5TextModel
        from transformers.utils import logging as hf_logging

        model_id = resolve_model_id(model_id)
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.frozen = False
        verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()  # the VL checkpoint's visual.* keys are skipped on purpose
        try:
            # eager attention: under the causal mask a left-padded row's pad queries see only masked
            # columns, for which the SDPA backward returns NaN in bf16
            if pretrained:
                self.model = Qwen3_5TextModel.from_pretrained(
                    model_id, dtype=torch.float32, attn_implementation="eager"
                )
            else:
                config = AutoConfig.from_pretrained(model_id)
                config = getattr(config, "text_config", config)
                self.model = Qwen3_5TextModel._from_config(config, dtype=torch.float32, attn_implementation="eager")
                logger.info("LLM backbone randomly initialised (pretrained: false)")
        finally:
            hf_logging.set_verbosity(verbosity)
        if num_layers is not None:
            _keep_first_layers(self.model, num_layers)

    def train(self, mode=True):
        return super().train(mode and not self.frozen)

    @property
    def hidden_size(self):
        return self.model.config.hidden_size

    @property
    def embed_scale(self):
        scale = getattr(self.model.get_input_embeddings(), "embed_scale", None)
        return float(scale.detach()) if scale is not None else 1.0

    @property
    def embed_rms(self):
        weight = self.model.get_input_embeddings().weight
        return float(weight.detach().float().std()) * self.embed_scale

    def embed_text(self, input_ids):
        return self.model.get_input_embeddings()(input_ids)

    def forward(self, inputs_embeds, attention_mask):
        if inputs_embeds.shape[0] == 1 and not bool(attention_mask.all()):
            raise ValueError("qwen3_5 linear attention ignores padding at batch size 1")
        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False)
        return out.last_hidden_state


def build_llm_backbone(cfg):
    """cfg: {kind: qwen3_5, model_id, num_layers (null = all), frozen, pretrained}."""
    cfg = dict(cfg)
    kind = cfg.pop("kind", "qwen3_5")
    if kind != "qwen3_5":
        raise ValueError(f"model.llm.kind must be qwen3_5 in this release, got {kind!r}")
    frozen = bool(cfg.pop("frozen", False))
    pretrained = bool(cfg.pop("pretrained", True))
    if not pretrained and frozen:
        raise ValueError("model.llm.pretrained: false requires frozen: false")
    backbone = Qwen35Backbone(cfg.pop("model_id"), num_layers=cfg.pop("num_layers", None), pretrained=pretrained)
    if cfg:
        raise ValueError(f"unknown model.llm keys: {sorted(cfg)}")
    if frozen:
        backbone.requires_grad_(False)
        backbone.frozen = True
        backbone.eval()
    return backbone
