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
    get_imagenet_labels,
    IMAGENET_MEAN,
    IMAGENET_STD
)
from src.attacks import SimBA, SquareAttack


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
    targeted: bool = False,
    opportunistic: bool = False
) -> Optional[Image.Image]:
    """Create a graph showing confidence evolution.

    Args:
        confidence_history: Dictionary with 'iterations', 'original_class', 'max_other_class',
                           and optionally 'target_class' keys.
        targeted: If True, show target class confidence line with "Target Class" label.
        opportunistic: If True, show locked class confidence line with "Locked Class" label.
                      (The locked class is the one chosen by the stability criterion.)

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

    # Add target/locked class line if available
    if opportunistic:
        # For opportunistic mode, reconstruct the locked class confidence from:
        # - top_classes data (before switch): look up locked_class in each dict
        # - target_class data (after switch): direct values
        locked_class = confidence_history.get('locked_class')
        top_classes = confidence_history.get('top_classes', [])
        target_conf_after = confidence_history.get('target_class', [])

        if locked_class is not None:
            # Build the full locked class confidence history
            locked_conf = []
            # Before switch: extract from top_classes
            for top_dict in top_classes:
                if locked_class in top_dict:
                    locked_conf.append(top_dict[locked_class])
                else:
                    # Class wasn't in top 10 at this point, use 0 or skip
                    locked_conf.append(None)
            # After switch: use target_class values
            locked_conf.extend(target_conf_after)

            # Filter out None values and align with iterations
            valid_indices = [i for i, c in enumerate(locked_conf) if c is not None]
            valid_conf = [locked_conf[i] for i in valid_indices]
            valid_iters = [iterations[i] for i in valid_indices if i < len(iterations)]

            if valid_conf and valid_iters:
                ax.plot(valid_iters[:len(valid_conf)], valid_conf[:len(valid_iters)],
                       'g-', label='Locked Class Confidence', linewidth=2)
    elif targeted and 'target_class' in confidence_history and len(confidence_history['target_class']) > 0:
        target_conf = confidence_history['target_class']
        # For targeted mode from the start, align with first N iterations
        target_iterations = iterations[:len(target_conf)]
        ax.plot(target_iterations, target_conf, 'g-', label='Target Class Confidence', linewidth=2)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Confidence', fontsize=12)
    if opportunistic:
        title = 'Confidence Evolution During Opportunistic Attack'
    elif targeted:
        title = 'Confidence Evolution During Targeted Attack'
    else:
        title = 'Confidence Evolution During Untargeted Attack'
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
    target_class: Optional[int] = None,
    opportunistic: bool = False,
    stability_threshold: int = 30,
    loss: str = 'margin'
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
        opportunistic: If True, start untargeted and switch to targeted when max class stabilizes.
        stability_threshold: Number of consecutive accepted perturbations before switching.
        loss: Loss function for Square Attack ('margin' or 'ce').

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
        elif method == "Square Attack":
            attack = SquareAttack(
                model=model,
                epsilon=epsilon,
                max_iterations=max_iterations,
                device=_device,
                loss=loss
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
            x_adv = attack.generate(
                image_tensor, y_true,
                track_confidence=True,
                opportunistic=opportunistic,
                stability_threshold=stability_threshold
            )
        
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

        # Get confidence history and check for opportunistic switch
        ch = getattr(attack, 'confidence_history', None)
        switch_iteration = ch.get('switch_iteration') if ch else None

        # Create confidence evolution graph
        # For opportunistic mode, only show locked class line if a switch occurred
        opportunistic_switched = opportunistic and switch_iteration is not None
        confidence_graph = create_confidence_graph(
            ch,
            targeted=targeted,
            opportunistic=opportunistic_switched
        )

        # Iterations used (from confidence history if available)
        iterations_used = ch['iterations'][-1] if ch and ch.get('iterations') else None
        budget_display = f"{iterations_used} iterations / {max_iterations}" if iterations_used is not None else f"? iterations / {max_iterations}"

        # Compute perturbation metrics
        total_perturbation = (x_adv - image_tensor).squeeze(0)
        linf = total_perturbation.abs().max().item()
        l2 = total_perturbation.norm(2).item()
        mean_l1 = total_perturbation.abs().mean().item()

        # Create result message
        perturbation_section = f"""**Perturbation Metrics:**
- L∞: {linf:.4f}
- L2: {l2:.4f}
- Mean L1: {mean_l1:.6f}"""

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

**Budget:** {budget_display}

{perturbation_section}

**Attack Status:** {'✓ Successful' if is_successful else '✗ Failed'}
"""
        elif opportunistic:
            mode_desc = "Opportunistic"
            if switch_iteration is not None:
                mode_desc += f" (switched to targeted at iteration {switch_iteration})"
            else:
                mode_desc += " (no switch occurred)"
            result_text = f"""**Attack Mode:** {mode_desc}

**Original Prediction:**
- Class: {original_class} ({original_label})
- Confidence: {original_confidence:.2%}

**Adversarial Prediction:**
- Class: {adv_class} ({adv_label})
- Confidence: {adv_confidence:.2%}

**Budget:** {budget_display}

{perturbation_section}

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

**Budget:** {budget_display}

{perturbation_section}

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
    
    # Build GPU status message
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_status_md = (
            f"<div style='display:inline-block; padding:4px 12px; "
            f"border-radius:6px; background:#d4edda; color:#155724; "
            f"font-size:0.9em'>"
            f"&#9679; GPU Active &mdash; <strong>{gpu_name}</strong></div>"
        )
    else:
        gpu_status_md = (
            "<div style='display:inline-block; padding:4px 12px; "
            "border-radius:6px; background:#e2e3e5; color:#6c757d; "
            "font-size:0.9em'>"
            "&#9675; GPU Unavailable &mdash; running on CPU</div>"
        )

    with gr.Blocks(title="Adversarial Attack Demonstrator") as demo:
        gr.Markdown(
            f"""
            # Adversarial Black-Box Attack Demonstrator

            Upload an image and configure the attack parameters to generate adversarial examples.

            {gpu_status_md}
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Attack Configuration")
                
                method_dropdown = gr.Dropdown(
                    choices=["SimBA", "Square Attack"],
                    value="Square Attack",
                    label="Attack Method",
                    info="Select the adversarial attack method"
                )
                
                epsilon_slider = gr.Slider(
                    minimum=0.01,
                    maximum=0.5,
                    value=0.03,
                    step=0.01,
                    label="Epsilon (ε)",
                    info="Maximum perturbation magnitude (L∞ norm)"
                )
                
                max_iter_slider = gr.Slider(
                    minimum=500,
                    maximum=10000,
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

                loss_radio = gr.Radio(
                    choices=["Cross-Entropy", "Margin"],
                    value="Cross-Entropy",
                    label="Loss Function",
                    info="Loss optimized by Square Attack",
                    visible=True
                )

                # Show/hide loss selector based on attack method
                def update_loss_visibility(method, current_loss):
                    return gr.Radio(
                        choices=["Cross-Entropy", "Margin"],
                        value=current_loss,
                        label="Loss Function",
                        info="Loss optimized by Square Attack",
                        visible=(method == "Square Attack"),
                    )

                method_dropdown.change(
                    fn=update_loss_visibility,
                    inputs=[method_dropdown, loss_radio],
                    outputs=[loss_radio]
                )

                gr.Markdown("### Attack Mode")

                attack_mode = gr.Radio(
                    choices=["Untargeted", "Targeted"],
                    value="Untargeted",
                    label="Attack Mode",
                    info="Untargeted: cause any misclassification. Targeted: force specific class."
                )

                imagenet_labels = get_imagenet_labels()

                target_class_number = gr.Number(
                    value=0,
                    label="Target Class Index (0-999)",
                    info="Enter an ImageNet class index",
                    precision=0,
                    visible=False
                )

                target_class_label = gr.Markdown(
                    value=f"**Target:** 0 — {imagenet_labels.get(0, 'unknown')}",
                    visible=False
                )

                opportunistic_checkbox = gr.Checkbox(
                    value=False,
                    label="Opportunistic Targeting",
                    info="Start untargeted, then switch to targeted when a class stabilizes",
                    visible=True
                )

                stability_threshold_slider = gr.Slider(
                    minimum=5,
                    maximum=50,
                    value=5,
                    step=5,
                    label="Stability Threshold",
                    info="Consecutive accepted perturbations before switching to targeted",
                    visible=False
                )

                # Show/hide controls based on attack mode
                def update_mode_visibility(mode):
                    is_targeted = (mode == "Targeted")
                    return (
                        gr.update(visible=is_targeted),  # target_class_number
                        gr.update(visible=is_targeted),  # target_class_label
                        gr.update(visible=not is_targeted),  # opportunistic_checkbox
                        gr.update(visible=False)  # stability_threshold_slider (controlled by checkbox)
                    )

                attack_mode.change(
                    fn=update_mode_visibility,
                    inputs=[attack_mode],
                    outputs=[target_class_number, target_class_label,
                             opportunistic_checkbox, stability_threshold_slider]
                )

                # Update label preview when class index changes
                def update_target_label(idx):
                    idx = int(idx) if idx is not None else 0
                    idx = max(0, min(999, idx))
                    name = imagenet_labels.get(idx, 'unknown')
                    return f"**Target:** {idx} — {name}"

                target_class_number.change(
                    fn=update_target_label,
                    inputs=[target_class_number],
                    outputs=[target_class_label]
                )

                # Show/hide stability threshold based on opportunistic checkbox
                def update_stability_visibility(opportunistic):
                    return gr.update(visible=opportunistic)

                opportunistic_checkbox.change(
                    fn=update_stability_visibility,
                    inputs=[opportunistic_checkbox],
                    outputs=[stability_threshold_slider]
                )

                attack_button = gr.Button("Run Attack", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                gr.Markdown("### Image")

                # Hidden component to store the raw uploaded image
                image_input = gr.Image(
                    type="pil",
                    visible=False
                )

                # Preprocessed image shown directly to the user
                original_output = gr.Image(
                    label="Preprocessed Image",
                    type="pil",
                    height=400,
                    sources=["upload", "clipboard"],
                    interactive=True
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

            with gr.Column(scale=1):
                gr.Markdown("### Results")

                confidence_graph_output = gr.Image(
                    label="Confidence Evolution",
                    type="pil",
                    height=300
                )

                result_text = gr.Markdown(label="Attack Results")
        
        # When user uploads directly to the visible component, store raw in hidden input
        def store_raw_image(image):
            return image

        original_output.upload(
            fn=store_raw_image,
            inputs=[original_output],
            outputs=[image_input]
        )

        # When hidden image_input changes (from upload or example), preprocess and show
        def update_original(image, model):
            if image is None:
                return None, ""
            try:
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

        # Only update prediction text when model changes (preprocessing is model-independent)
        def update_prediction(image, model):
            if image is None:
                return ""
            try:
                label, confidence, _ = predict_image(image, model)
                return f"**Original Prediction:** {label} ({confidence:.2%})"
            except Exception as e:
                return f"Error: {str(e)}"

        model_dropdown.change(
            fn=update_prediction,
            inputs=[image_input, model_dropdown],
            outputs=[result_text]
        )
        
        # Run attack when button is clicked
        # Note: original_output is NOT in outputs - it stays static during attack
        def execute_attack(image, method, epsilon, max_iter, model, loss_choice,
                          mode, target_cls_idx, opportunistic, stability_threshold):
            if image is None:
                return None, None, None, "Please upload an image first."

            targeted = (mode == "Targeted")
            target_class = max(0, min(999, int(target_cls_idx))) if targeted and target_cls_idx is not None else None

            # Opportunistic only applies to untargeted mode
            use_opportunistic = opportunistic and not targeted

            # Map UI label to torchattacks loss name
            loss = 'margin' if loss_choice == "Margin" else 'ce'

            adv_image, pert_image, conf_graph, result = run_attack(
                image, method, epsilon, max_iter, model,
                targeted=targeted, target_class=target_class,
                opportunistic=use_opportunistic, stability_threshold=int(stability_threshold),
                loss=loss
            )
            return adv_image, pert_image, conf_graph, result

        attack_button.click(
            fn=execute_attack,
            inputs=[image_input, method_dropdown, epsilon_slider, max_iter_slider,
                    model_dropdown, loss_radio, attack_mode, target_class_number,
                    opportunistic_checkbox, stability_threshold_slider],
            outputs=[adversarial_output, perturbation_output, confidence_graph_output, result_text]
        )
        
        # Example images
        gr.Markdown("### Example Images")
        example_images = []
        example_dir = os.path.join(project_root, "data")
        if os.path.exists(example_dir):
            for filename in os.listdir(example_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
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
    demo.launch(share=share, server_name=server_name, server_port=server_port, theme=gr.themes.Soft())


if __name__ == "__main__":
    launch_demo()
