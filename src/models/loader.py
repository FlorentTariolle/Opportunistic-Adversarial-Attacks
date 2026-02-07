"""Model loading utilities for adversarial attack demonstrations."""

import warnings
from typing import Optional, Union
import torch
import torch.nn as nn
import torchvision.models as models


# Curated RobustBench ImageNet Linf models
ROBUSTBENCH_MODELS = {
    'Salman2020Do_R18': 'Salman2020Do_R18',
    'Salman2020Do_R50': 'Salman2020Do_R50',
    'Wong2020Fast': 'Wong2020Fast',
    'Engstrom2019Robustness': 'Engstrom2019Robustness',
    'Debenedetti2022Light_XCiT-S12': 'Debenedetti2022Light_XCiT-S12',
}


def load_pretrained_model(
    model_name: str = 'resnet18',
    num_classes: Optional[int] = None,
    pretrained: bool = True,
    device: Optional[torch.device] = None
) -> nn.Module:
    """Load a pretrained model from torchvision.
    
    Args:
        model_name: Name of the model to load (e.g., 'resnet18', 'vgg16').
        num_classes: Number of classes. If None, uses default (usually 1000).
        pretrained: Whether to load pretrained weights.
        device: Device to load the model on. If None, uses CPU.
    
    Returns:
        Loaded model in eval mode.
    
    Raises:
        ValueError: If model_name is not supported.
    """
    if device is None:
        device = torch.device('cpu')
    
    # Map of supported model names to their constructors
    model_constructors = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'vgg16': models.vgg16,
        'vgg19': models.vgg19,
        'alexnet': models.alexnet,
        'mobilenet_v2': models.mobilenet_v2,
        'densenet121': models.densenet121,
    }
    
    if model_name not in model_constructors:
        raise ValueError(
            f"Model '{model_name}' not supported. "
            f"Supported models: {list(model_constructors.keys())}"
        )
    
    # Load model
    # Use weights parameter instead of deprecated pretrained parameter
    if pretrained:
        # Try to use weights parameter (torchvision >= 0.13)
        try:
            # Use DEFAULT weights which is equivalent to pretrained=True
            model = model_constructors[model_name](weights="DEFAULT")
        except (TypeError, ValueError):
            # Fallback for older torchvision versions that use pretrained parameter
            model = model_constructors[model_name](pretrained=True)
    else:
        try:
            model = model_constructors[model_name](weights=None)
        except (TypeError, ValueError):
            model = model_constructors[model_name](pretrained=False)
        if num_classes is None:
            num_classes = 1000
    
    # Modify number of classes if specified
    if num_classes is not None and num_classes != 1000:
        if hasattr(model, 'fc'):
            # ResNet, DenseNet, etc.
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif hasattr(model, 'classifier'):
            # VGG, AlexNet, etc.
            if isinstance(model.classifier, nn.Sequential):
                model.classifier[-1] = nn.Linear(
                    model.classifier[-1].in_features, num_classes
                )
            else:
                model.classifier = nn.Linear(
                    model.classifier.in_features, num_classes
                )
    
    model = model.to(device)
    model.eval()
    
    return model


def load_robustbench_model(
    model_name: str,
    device: Optional[torch.device] = None
) -> nn.Module:
    """Load an adversarially-trained model from RobustBench.

    Uses lazy import to avoid slow timm/robustbench startup overhead.

    Args:
        model_name: RobustBench model identifier (e.g. 'Salman2020Do_R50').
        device: Device to load the model on. If None, uses CPU.

    Returns:
        Loaded model in eval mode. Expects [0, 1] input (has built-in normalizer).

    Raises:
        ValueError: If model_name is not in ROBUSTBENCH_MODELS.
    """
    if model_name not in ROBUSTBENCH_MODELS:
        raise ValueError(
            f"RobustBench model '{model_name}' not supported. "
            f"Supported: {list(ROBUSTBENCH_MODELS.keys())}"
        )

    if device is None:
        device = torch.device('cpu')

    from robustbench.utils import load_model
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
        model = load_model(
            model_name=ROBUSTBENCH_MODELS[model_name],
            dataset='imagenet',
            threat_model='Linf',
        )
    model = model.to(device)
    model.eval()
    return model


def get_model(
    model_name: str = 'resnet18',
    device: Optional[torch.device] = None,
    source: str = 'standard',
) -> nn.Module:
    """Convenience function to get a pretrained model.

    Args:
        model_name: Name of the model to load.
        device: Device to load the model on.
        source: 'standard' for torchvision models, 'robust' for RobustBench.

    Returns:
        Loaded model in eval mode.
    """
    if source == 'robust':
        return load_robustbench_model(model_name=model_name, device=device)
    return load_pretrained_model(model_name=model_name, device=device)
