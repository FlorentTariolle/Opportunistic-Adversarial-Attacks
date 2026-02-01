"""Image preprocessing and visualization utilities."""

from typing import Optional, Tuple, Union, Dict
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
import os


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_imagenet_labels() -> Dict[int, str]:
    """Get ImageNet class index to label mapping.

    Returns:
        Dictionary mapping class index (int) to human-readable label (str).

    Raises:
        FileNotFoundError: If the imagenet_class_index.json file is not found.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    json_path = os.path.join(project_root, 'data', 'imagenet_class_index.json')

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"ImageNet class index not found at {json_path}.")

    with open(json_path, 'r') as f:
        class_index = json.load(f)
        # Convert from {"0": ["n01440764", "tench"], ...} to {0: "tench", ...}
        return {int(k): v[1] for k, v in class_index.items()}


def get_imagenet_label(class_idx: int) -> str:
    """Get human-readable label for an ImageNet class index.
    
    Args:
        class_idx: ImageNet class index (0-999).
    
    Returns:
        Human-readable label string.
    """
    labels = get_imagenet_labels()
    return labels.get(class_idx, f"class_{class_idx}")


def load_image(
    path: str,
    size: Optional[Tuple[int, int]] = (224, 224)
) -> Image.Image:
    """Load an image from file.
    
    Args:
        path: Path to the image file.
        size: Optional tuple (width, height) to resize the image.
    
    Returns:
        PIL Image object.
    
    Raises:
        FileNotFoundError: If the image file doesn't exist.
        IOError: If the image cannot be opened.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    
    try:
        image = Image.open(path).convert('RGB')
    except Exception as e:
        raise IOError(f"Failed to open image {path}: {e}")
    
    if size is not None:
        image = image.resize(size)
    return image


def preprocess_image(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    normalize: bool = True,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Preprocess an image for model input.
    
    Args:
        image: Input image (PIL Image, numpy array, or torch Tensor).
        normalize: Whether to apply ImageNet normalization.
        device: Device to move tensor to.
    
    Returns:
        Preprocessed tensor of shape (1, 3, H, W) ready for model input.
    """
    if device is None:
        device = torch.device('cpu')
    
    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    elif isinstance(image, torch.Tensor):
        # Assume tensor is in [0, 1] range
        image = transforms.ToPILImage()(image.cpu())
    
    # Define transforms
    transform_list = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
    
    if normalize:
        transform_list.append(
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        )
    
    transform = transforms.Compose(transform_list)
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return tensor.to(device)


def normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet normalization to a tensor.
    
    Args:
        x: Input tensor of shape (..., 3, H, W) with values in [0, 1].
    
    Returns:
        Normalized tensor.
    """
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def denormalize_image(
    x: torch.Tensor,
    mean: Optional[list] = None,
    std: Optional[list] = None
) -> torch.Tensor:
    """Denormalize an image tensor.
    
    Args:
        x: Normalized tensor of shape (..., 3, H, W).
        mean: Mean values used for normalization. Defaults to ImageNet mean.
        std: Std values used for normalization. Defaults to ImageNet std.
    
    Returns:
        Denormalized tensor with values in [0, 1] range.
    """
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD
    
    mean_tensor = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    
    x_denorm = x * std_tensor + mean_tensor
    return torch.clamp(x_denorm, 0.0, 1.0)


def show_image(
    image: Union[torch.Tensor, np.ndarray, Image.Image],
    title: Optional[str] = None,
    denormalize: bool = False,
    figsize: Tuple[int, int] = (8, 8)
) -> None:
    """Display an image.
    
    Args:
        image: Image to display (tensor, numpy array, or PIL Image).
        title: Optional title for the plot.
        denormalize: If True and image is a tensor, apply denormalization.
        figsize: Figure size tuple (width, height).
    """
    # Convert tensor to numpy
    if isinstance(image, torch.Tensor):
        if denormalize:
            image = denormalize_image(image)
        
        # Remove batch dimension if present
        if image.dim() == 4:
            image = image[0]
        
        # Convert to numpy and transpose if needed
        image = image.cpu().detach().numpy()
        if image.shape[0] == 3:  # CHW format
            image = np.transpose(image, (1, 2, 0))
    
    # Convert PIL to numpy
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure values are in [0, 1] range
    if image.max() > 1.0:
        image = image / 255.0
    
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()
