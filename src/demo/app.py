"""Gradio-based demonstrator for adversarial attacks."""

import os
import sys
import io
from typing import Optional, Tuple, Dict
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import gradio as gr
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.loader import get_model
from src.utils.imaging import (
    preprocess_image, 
    denormalize_image, 
    get_imagenet_label,
    IMAGENET_MEAN,
    IMAGENET_STD
)
from src.attacks import SimBA


# Global model cache
_model_cache = {}
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_cached_model(model_name: str = 'resnet18'):
    """Get or load a cached model.
    
    Args:
        model_name: Name of the model to load (e.g., 'resnet18', 'vgg16').
    
    Returns:
        Loaded model in eval mode.
    """
    if model_name not in _model_cache:
        _model_cache[model_name] = get_model(model_name=model_name, device=_device)
    return _model_cache[model_name]


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a tensor to PIL Image.
    
    Args:
        tensor: Input tensor of shape (C, H, W) or (1, C, H, W), 
                optionally normalized (values may be negative).
    
    Returns:
        PIL Image with values in [0, 255] range.
    """
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # Denormalize if needed
    if tensor.min() < 0:  # Likely normalized
        tensor = denormalize_image(tensor.unsqueeze(0)).squeeze(0)
    
    # Convert to numpy and transpose if needed
    if tensor.shape[0] == 3:  # CHW format
        tensor = tensor.permute(1, 2, 0)
    
    # Clamp to [0, 1] and convert to uint8
    tensor = torch.clamp(tensor, 0.0, 1.0)
    numpy_array = (tensor.cpu().detach().numpy() * 255).astype(np.uint8)
    
    return Image.fromarray(numpy_array)


def compute_perturbation_visualization(
    original: torch.Tensor,
    adversarial: torch.Tensor
) -> Image.Image:
    """Compute and visualize perturbation (scaled for visibility).
    
    Args:
        original: Original image tensor (normalized).
        adversarial: Adversarial image tensor (normalized).
    
    Returns:
        PIL Image showing the perturbation scaled for visibility.
    """
    # Denormalize both images
    orig_denorm = denormalize_image(original)
    adv_denorm = denormalize_image(adversarial)
    
    # Compute perturbation
    perturbation = adv_denorm - orig_denorm
    
    # Remove batch dimension if present
    if perturbation.dim() == 4:
        perturbation = perturbation[0]
    
    # Convert to numpy and transpose if needed
    if perturbation.shape[0] == 3:  # CHW format
        perturbation = perturbation.permute(1, 2, 0)
    
    perturbation_np = perturbation.cpu().detach().numpy()
    
    # Scale perturbation for visibility: shift to [0, 1] range
    # Perturbations are typically in [-epsilon, epsilon], so we shift and scale
    perturbation_min = perturbation_np.min()
    perturbation_max = perturbation_np.max()
    
    if perturbation_max > perturbation_min:
        # Scale to [0, 1]
        perturbation_scaled = (perturbation_np - perturbation_min) / (perturbation_max - perturbation_min)
    else:
        perturbation_scaled = np.zeros_like(perturbation_np)
    
    # Convert to uint8
    perturbation_uint8 = (perturbation_scaled * 255).astype(np.uint8)
    
    return Image.fromarray(perturbation_uint8)


def create_confidence_graph(
    confidence_history: Optional[Dict],
    targeted: bool = False
) -> Optional[Image.Image]:
    """Create a graph showing confidence evolution.

    Args:
        confidence_history: Dictionary with 'iterations', 'original_class', 'max_other_class',
                           and optionally 'target_class' keys.
        targeted: If True, show target class confidence line.

    Returns:
        PIL Image of the graph, or None if no history available.
    """
    if confidence_history is None or len(confidence_history['iterations']) == 0:
        return None

    iterations = confidence_history['iterations']
    original_conf = confidence_history['original_class']
    max_other_conf = confidence_history['max_other_class']

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iterations, original_conf, 'b-', label='Original Class Confidence', linewidth=2)
    ax.plot(iterations, max_other_conf, 'r--', label='Max Other Class Confidence', linewidth=2)

    # Add target class line for targeted attacks
    if targeted and 'target_class' in confidence_history and len(confidence_history['target_class']) > 0:
        target_conf = confidence_history['target_class']
        ax.plot(iterations[:len(target_conf)], target_conf, 'g-', label='Target Class Confidence', linewidth=2)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Confidence', fontsize=12)
    title = 'Confidence Evolution During Targeted Attack' if targeted else 'Confidence Evolution During Attack'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()

    # Convert to PIL Image using BytesIO
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img


def predict_image(image: Image.Image, model_name: str = 'resnet18') -> Tuple[str, float, int]:
    """Get model prediction for an image.
    
    Args:
        image: Input PIL Image.
        model_name: Name of the model to use for prediction.
    
    Returns:
        Tuple of (label, confidence, class_index) where:
        - label: Human-readable class label
        - confidence: Prediction confidence (0-1)
        - class_index: ImageNet class index
    """
    model = get_cached_model(model_name)
    
    # Preprocess image
    image_tensor = preprocess_image(image, device=_device)
    
    # Get prediction
    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = probs[0][predicted_class].item()
    
    label = get_imagenet_label(predicted_class)
    return label, confidence, predicted_class


def run_attack(
    image: Optional[Image.Image],
    method: str,
    epsilon: float,
    max_iterations: int,
    model_name: str = 'resnet18',
    targeted: bool = False,
    target_class: Optional[int] = None
) -> Tuple[Image.Image, Image.Image, Image.Image, str]:
    """Run adversarial attack on the image.

    Args:
        image: Input PIL Image to attack.
        method: Attack method name (e.g., 'SimBA').
        epsilon: Maximum perturbation magnitude (L∞ norm).
        max_iterations: Maximum number of attack iterations.
        model_name: Name of the target model to attack.
        targeted: If True, perform targeted attack.
        target_class: Target class index for targeted attack.

    Returns:
        Tuple of (adversarial_image, perturbation_image, confidence_graph, result_text):
        - adversarial_image: Generated adversarial example
        - perturbation_image: Visualization of the perturbation (scaled)
        - confidence_graph: Graph showing confidence evolution during attack
        - result_text: Formatted text with attack results
    """
    if image is None:
        return None, None, None, "Please upload an image first."
    
    try:
        # Load model
        model = get_cached_model(model_name)
        
        # Preprocess image
        image_tensor = preprocess_image(image, device=_device)
        
        # Get original prediction
        with torch.no_grad():
            logits = model(image_tensor)
            probs = F.softmax(logits, dim=1)
            original_class = torch.argmax(logits, dim=1).item()
            original_confidence = probs[0][original_class].item()
        
        original_label = get_imagenet_label(original_class)
        
        # Initialize attack based on method
        if method == "SimBA":
            attack = SimBA(
                model=model,
                epsilon=epsilon,
                max_iterations=max_iterations,
                device=_device,
                use_dct=True
            )
        else:
            return None, None, None, f"Unknown attack method: {method}"
        
        # Run attack with confidence tracking
        y_true = torch.tensor([original_class], device=_device)
        if targeted and target_class is not None:
            target_tensor = torch.tensor([target_class], device=_device)
            x_adv = attack.generate(
                image_tensor, y_true,
                track_confidence=True,
                targeted=True,
                target_class=target_tensor
            )
        else:
            x_adv = attack.generate(image_tensor, y_true, track_confidence=True)
        
        # Get adversarial prediction
        with torch.no_grad():
            adv_logits = model(x_adv)
            adv_probs = F.softmax(adv_logits, dim=1)
            adv_class = torch.argmax(adv_logits, dim=1).item()
            adv_confidence = adv_probs[0][adv_class].item()
        
        adv_label = get_imagenet_label(adv_class)

        # Check if attack was successful
        if targeted and target_class is not None:
            is_successful = (adv_class == target_class)
            target_label = get_imagenet_label(target_class)
        else:
            is_successful = attack.check_adversarial(x_adv, y_true).item()

        # Convert adversarial tensor to PIL
        adv_image = tensor_to_pil(x_adv)

        # Compute perturbation visualization
        perturbation_image = compute_perturbation_visualization(image_tensor, x_adv)

        # Create confidence evolution graph
        confidence_graph = create_confidence_graph(
            getattr(attack, 'confidence_history', None),
            targeted=targeted
        )

        # Create result message
        if targeted and target_class is not None:
            result_text = f"""**Attack Mode:** Targeted

**Original Prediction:**
- Class: {original_class} ({original_label})
- Confidence: {original_confidence:.2%}

**Target Class:**
- Class: {target_class} ({target_label})

**Adversarial Prediction:**
- Class: {adv_class} ({adv_label})
- Confidence: {adv_confidence:.2%}

**Attack Status:** {'✓ Successful' if is_successful else '✗ Failed'}
"""
        else:
            result_text = f"""**Attack Mode:** Untargeted

**Original Prediction:**
- Class: {original_class} ({original_label})
- Confidence: {original_confidence:.2%}

**Adversarial Prediction:**
- Class: {adv_class} ({adv_label})
- Confidence: {adv_confidence:.2%}

**Attack Status:** {'✓ Successful' if is_successful else '✗ Failed'}
"""
        
        return adv_image, perturbation_image, confidence_graph, result_text
        
    except Exception as e:
        import traceback
        error_msg = f"Error during attack: {str(e)}\n\n{traceback.format_exc()}"
        return None, None, None, error_msg


def create_demo_interface():
    """Create the Gradio interface for the adversarial attack demonstrator.
    
    Returns:
        Gradio Blocks interface with image upload, attack configuration,
        and visualization components.
    """
    
    with gr.Blocks(title="Adversarial Attack Demonstrator", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # Adversarial Black-Box Attack Demonstrator
            
            Upload an image and configure the attack parameters to generate adversarial examples.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Attack Configuration")
                
                method_dropdown = gr.Dropdown(
                    choices=["SimBA"],
                    value="SimBA",
                    label="Attack Method",
                    info="Select the adversarial attack method"
                )
                
                epsilon_slider = gr.Slider(
                    minimum=0.01,
                    maximum=1.0,
                    value=0.03,
                    step=0.01,
                    label="Epsilon (ε)",
                    info="Maximum perturbation magnitude (L∞ norm)"
                )
                
                max_iter_slider = gr.Slider(
                    minimum=100,
                    maximum=5000,
                    value=1000,
                    step=100,
                    label="Max Iterations",
                    info="Maximum number of attack iterations"
                )
                
                model_dropdown = gr.Dropdown(
                    choices=["resnet18", "resnet34", "resnet50", "vgg16", "vgg19", "alexnet"],
                    value="resnet18",
                    label="Model",
                    info="Target model to attack"
                )

                gr.Markdown("### Attack Mode")

                attack_mode = gr.Radio(
                    choices=["Untargeted", "Targeted"],
                    value="Untargeted",
                    label="Attack Mode",
                    info="Untargeted: cause any misclassification. Targeted: force specific class."
                )

                target_class_input = gr.Number(
                    value=0,
                    label="Target Class (0-999)",
                    info="ImageNet class index for targeted attack",
                    minimum=0,
                    maximum=999,
                    step=1,
                    visible=False
                )

                # Show/hide target class input based on attack mode
                def update_target_visibility(mode):
                    return gr.update(visible=(mode == "Targeted"))

                attack_mode.change(
                    fn=update_target_visibility,
                    inputs=[attack_mode],
                    outputs=[target_class_input]
                )

                attack_button = gr.Button("Run Attack", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                gr.Markdown("### 📸 Image")
                
                image_input = gr.Image(
                    type="pil",
                    label="Upload Image",
                    height=400
                )
                
                # Store preprocessed original image tensor (for immediate display)
                original_output = gr.Image(
                    label="Original Image",
                    type="pil",
                    height=300
                )
                
                with gr.Row():
                    adversarial_output = gr.Image(
                        label="Adversarial Image",
                        type="pil",
                        height=300
                    )
                    
                    perturbation_output = gr.Image(
                        label="Perturbation (scaled for visibility)",
                        type="pil",
                        height=300
                    )
                
                confidence_graph_output = gr.Image(
                    label="Confidence Evolution",
                    type="pil",
                    height=300
                )
                
                result_text = gr.Markdown(label="Attack Results")
        
        # Update original image immediately when input changes (preprocess and show)
        def update_original(image, model):
            if image is None:
                return None, ""
            try:
                # Preprocess and show immediately
                preprocessed = preprocess_image(image, device=_device)
                original_pil = tensor_to_pil(preprocessed)
                label, confidence, _ = predict_image(image, model)
                return original_pil, f"**Original Prediction:** {label} ({confidence:.2%})"
            except Exception as e:
                return image, f"Error: {str(e)}"
        
        image_input.change(
            fn=update_original,
            inputs=[image_input, model_dropdown],
            outputs=[original_output, result_text]
        )
        
        # Also update when model changes
        model_dropdown.change(
            fn=update_original,
            inputs=[image_input, model_dropdown],
            outputs=[original_output, result_text]
        )
        
        # Run attack when button is clicked
        # Note: original_output is NOT in outputs - it stays static during attack
        def execute_attack(image, method, epsilon, max_iter, model, mode, target_cls):
            if image is None:
                return None, None, None, "Please upload an image first."

            targeted = (mode == "Targeted")
            target_class = int(target_cls) if targeted else None

            adv_image, pert_image, conf_graph, result = run_attack(
                image, method, epsilon, max_iter, model,
                targeted=targeted, target_class=target_class
            )
            return adv_image, pert_image, conf_graph, result

        attack_button.click(
            fn=execute_attack,
            inputs=[image_input, method_dropdown, epsilon_slider, max_iter_slider,
                    model_dropdown, attack_mode, target_class_input],
            outputs=[adversarial_output, perturbation_output, confidence_graph_output, result_text]
        )
        
        # Example images
        gr.Markdown("### 📁 Example Images")
        example_images = []
        example_dir = os.path.join(project_root, "data")
        if os.path.exists(example_dir):
            for filename in os.listdir(example_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    example_images.append(os.path.join(example_dir, filename))
        
        if example_images:
            gr.Examples(
                examples=example_images,
                inputs=image_input,
                label="Click to load example"
            )
    
    return demo


def launch_demo(share: bool = False, server_name: str = "127.0.0.1", server_port: int = 7860):
    """Launch the Gradio demonstrator.
    
    Args:
        share: If True, creates a public link for sharing.
        server_name: Server name to bind to.
        server_port: Port to run the server on.
    """
    demo = create_demo_interface()
    demo.launch(share=share, server_name=server_name, server_port=server_port)


if __name__ == "__main__":
    launch_demo()
