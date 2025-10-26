import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
import math
from colorama import Fore
from utils.logger import get_logger
from utils.rich_handlers import ModelHandler
from torchinfo import summary


def _get_1d_sincos_pos_embed(length: int, dim: int, temperature: float = 10000.0, device=None):
    """Generate 1D sine-cosine positional embedding."""
    assert dim % 2 == 0
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(temperature) / dim)
    )
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def build_2d_sincos_position_embedding(height: int, width: int, dim: int, device=None):
    """Create 2D sine-cos positional encoding of shape (1, H*W, dim)."""
    assert dim % 2 == 0, "positional dim must be even"
    dim_half = dim // 2
    pe_y = _get_1d_sincos_pos_embed(height, dim_half, device=device)
    pe_x = _get_1d_sincos_pos_embed(width, dim_half, device=device)

    pos = torch.zeros(height, width, dim, device=device, dtype=torch.float32)
    pos[:, :, :dim_half] = pe_y[:, None, :].expand(-1, width, -1)
    pos[:, :, dim_half:] = pe_x[None, :, :].expand(height, -1, -1)
    pos = pos.view(1, height * width, dim)
    return pos


class DETR(nn.Module):
    def __init__(
        self,
        num_classes,
        hidden_dim=256,
        nheads=8,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_queries=25,
    ):
        super().__init__()

        # Initialize logger and model handler
        self.logger = get_logger("model")
        self.model_handler = ModelHandler()

        # Log model configuration
        model_config = {
            "Model Type": "DETR (Detection Transformer)",
            "Number of Classes": num_classes,
            "Hidden Dimension": hidden_dim,
            "Attention Heads": nheads,
            "Encoder Layers": num_encoder_layers,
            "Decoder Layers": num_decoder_layers,
            "Object Queries": num_queries,
            "Backbone": "ResNet-50 (ImageNet pretrained)",
        }
        self.model_handler.log_model_architecture(model_config)

        # Backbone (ResNet-50)
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()

        # Conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # Transformer
        self.transformer = nn.Transformer(
            hidden_dim,
            nheads,
            num_encoder_layers,
            num_decoder_layers,
            batch_first=True,
            dropout=0.1,
        )

        # Prediction heads (extra class for "no object")
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # Queries and positional encodings
        self.num_queries = num_queries
        self.query_pos = nn.Parameter(torch.randn(self.num_queries, hidden_dim))

        # Normalizations
        self.norm_src = nn.LayerNorm(hidden_dim)
        self.norm_tgt = nn.LayerNorm(hidden_dim)

    def forward(self, inputs):
        """Forward pass through DETR."""
        # Pass through ResNet backbone
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # Convert from 2048 to hidden_dim feature planes
        feat = self.conv(x)
        bsz, d_model, Hf, Wf = feat.shape
        src = feat.flatten(2).permute(0, 2, 1)

        # Positional encodings
        pos = build_2d_sincos_position_embedding(Hf, Wf, d_model, device=feat.device)
        src = self.norm_src(src + pos)

        # Decoder targets
        tgt = torch.zeros(bsz, self.num_queries, d_model, device=feat.device)
        query_pos = self.query_pos.unsqueeze(0).expand(bsz, -1, -1)
        tgt = self.norm_tgt(tgt + query_pos)

        # Transformer
        hs = self.transformer(src=src, tgt=tgt)

        # Predictions
        return {
            "pred_logits": self.linear_class(hs),
            "pred_boxes": self.linear_bbox(hs).sigmoid(),
        }

    def log_model_info(self):
        """Log model parameter info."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.model_handler.log_parameters_count(total_params, trainable_params)

    def load_pretrained(self, checkpoint_path: str):
        """Load pretrained weights with compatibility for PyTorch 2.6+."""
        try:
            # Attempt default load (weights_only=True by default)
            checkpoint = torch.load(checkpoint_path)
        except Exception as e:
            self.logger.error(
                f"Default torch.load failed ({str(e)}). Retrying with weights_only=False..."
            )
            try:
                checkpoint = torch.load(checkpoint_path, weights_only=False)
            except Exception as e2:
                self.logger.error(f"Failed to load checkpoint after retry: {str(e2)}")
                self.model_handler.log_model_loading(checkpoint_path, success=False)
                return

        # Handle both checkpoint structures
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        try:
            self.load_state_dict(state_dict)
            self.model_handler.log_model_loading(checkpoint_path, success=True)
            self.logger.realtime(
                f"✅ Successfully loaded pretrained weights from {checkpoint_path}"
            )
        except Exception as e:
            self.logger.error(f"Failed to apply loaded weights: {str(e)}")
            self.model_handler.log_model_loading(checkpoint_path, success=False)


if __name__ == "__main__":
    model = DETR(num_classes=3)
    summary(model, (5, 3, 224, 224))
