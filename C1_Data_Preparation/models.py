"""C1 model initialization. Heads output nine disease classes."""
import timm
MODEL_NAMES = {
    "FasterNet": "fasternet_t0.in1k",
    "EfficientViT": "efficientvit_b0.r224_in1k",
    "GhostNet": "ghostnet_100.in1k",
}
def initialize_model(architecture: str, num_classes: int = 9, pretrained: bool = True):
    if architecture not in MODEL_NAMES:
        raise ValueError(f"Unknown architecture {architecture}; choose {list(MODEL_NAMES)}")
    return timm.create_model(MODEL_NAMES[architecture], pretrained=pretrained, num_classes=num_classes)
