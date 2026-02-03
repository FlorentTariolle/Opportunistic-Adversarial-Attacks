"""
Mini-benchmark: Targeted direct vs. Hybrid (100 untargeted + targeted)

Compare deux approches pour atteindre la classe cible 285 :
1. Direct : Attaque targeted directement
2. Hybride : 100 itérations untargeted puis targeted depuis l'image perturbée

Paramètres :
- Epsilon : 0.5, 1.25, 2.0
- Seeds : 3 par configuration
- Budget max : 2000 itérations
- Critère de succès : classe 285 devient top-1
"""

import torch
import numpy as np
import random
from src.models.loader import get_model
from src.utils.imaging import load_image, preprocess_image, get_imagenet_label
from src.attacks import SimBA


def set_seed(seed: int):
    """Fixe toutes les seeds pour reproductibilité."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_targeted_attack(model, image_tensor, true_class, target_class, epsilon, max_iter, device):
    """
    Lance une attaque targeted et retourne le nombre d'itérations.
    Retourne (iterations, success).
    """
    attack = SimBA(
        model=model,
        epsilon=epsilon,
        max_iterations=max_iter,
        device=device,
        use_dct=True
    )

    y_true = torch.tensor([true_class], device=device)
    target_tensor = torch.tensor([target_class], device=device)

    x_adv = attack.generate(
        image_tensor.clone(),
        y_true,
        track_confidence=True,
        targeted=True,
        target_class=target_tensor
    )

    # Vérifier le succès
    with torch.no_grad():
        adv_logits = model(x_adv)
        adv_class = torch.argmax(adv_logits, dim=1).item()

    success = (adv_class == target_class)

    # Récupérer le nombre d'itérations effectuées (en échec = budget max)
    if attack.confidence_history and attack.confidence_history['iterations']:
        iterations = attack.confidence_history['iterations'][-1]
    else:
        iterations = 0
    if not success:
        iterations = max_iter

    return iterations, success


def run_untargeted_attack(model, image_tensor, true_class, epsilon, max_iter, device, fixed_iterations=False):
    """
    Lance une attaque untargeted et retourne l'image perturbée.
    Si fixed_iterations=True, exécute exactement max_iter itérations (pas d'arrêt anticipé).
    Retourne (x_adv, iterations).
    """
    attack = SimBA(
        model=model,
        epsilon=epsilon,
        max_iterations=max_iter,
        device=device,
        use_dct=True
    )

    y_true = torch.tensor([true_class], device=device)

    x_adv = attack.generate(
        image_tensor.clone(),
        y_true,
        track_confidence=True,
        targeted=False,
        early_stop=not fixed_iterations
    )

    # Récupérer le nombre d'itérations effectuées
    if attack.confidence_history and attack.confidence_history['iterations']:
        iterations = attack.confidence_history['iterations'][-1]
    else:
        iterations = max_iter

    return x_adv, iterations


def run_hybrid_attack(model, image_tensor, true_class, target_class, epsilon, untargeted_iter, max_targeted_iter, device):
    """
    Lance 100 itérations untargeted puis targeted depuis l'image perturbée.
    Retourne (total_iterations, success).
    """
    # Phase 1 : Untargeted (exactement untargeted_iter itérations, pas d'arrêt anticipé)
    x_perturbed, untargeted_done = run_untargeted_attack(
        model, image_tensor, true_class, epsilon, untargeted_iter, device,
        fixed_iterations=True
    )

    # Phase 2 : Targeted depuis l'image perturbée
    # On doit récupérer la classe actuelle de l'image perturbée pour y_true
    with torch.no_grad():
        perturbed_logits = model(x_perturbed)
        current_class = torch.argmax(perturbed_logits, dim=1).item()

    # Vérifier si on a déjà atteint la cible par hasard
    if current_class == target_class:
        return untargeted_iter, True

    targeted_iter, success = run_targeted_attack(
        model, x_perturbed, current_class, target_class, epsilon, max_targeted_iter, device
    )

    # Total = 100 untargeted + iterations targeted
    total_iterations = untargeted_iter + targeted_iter

    return total_iterations, success


def main():
    # Configuration
    TARGET_CLASS = 285
    EPSILONS = [0.5, 1.25, 2.0]
    SEEDS = [42, 123, 456]
    MAX_ITERATIONS = 2000
    UNTARGETED_ITERATIONS = 100
    IMAGE_PATH = "data/cat.jpg"

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Target class: {TARGET_CLASS} ({get_imagenet_label(TARGET_CLASS)})")
    print(f"Epsilons: {EPSILONS}")
    print(f"Seeds: {SEEDS}")
    print(f"Max iterations: {MAX_ITERATIONS}")
    print(f"Untargeted warmup: {UNTARGETED_ITERATIONS} iterations")
    print("=" * 70)

    # Charger le modèle
    model = get_model('resnet18', device=device)

    # Charger l'image
    image = load_image(IMAGE_PATH)
    image_tensor = preprocess_image(image, normalize=True, device=device)

    # Obtenir la classe originale
    with torch.no_grad():
        logits = model(image_tensor)
        original_class = torch.argmax(logits, dim=1).item()

    print(f"Original class: {original_class} ({get_imagenet_label(original_class)})")
    print("=" * 70)

    # Résultats
    results = {eps: {'direct': [], 'hybrid': []} for eps in EPSILONS}

    for epsilon in EPSILONS:
        print(f"\n{'='*70}")
        print(f"EPSILON = {epsilon}")
        print(f"{'='*70}")

        for seed in SEEDS:
            print(f"\n--- Seed {seed} ---")

            # Approche directe
            set_seed(seed)
            direct_iter, direct_success = run_targeted_attack(
                model, image_tensor, original_class, TARGET_CLASS,
                epsilon, MAX_ITERATIONS, device
            )
            results[epsilon]['direct'].append((direct_iter, direct_success))
            status_direct = "OK" if direct_success else "FAIL"
            print(f"Direct targeted:  {direct_iter:4d} iterations [{status_direct}]")

            # Approche hybride
            set_seed(seed)
            hybrid_iter, hybrid_success = run_hybrid_attack(
                model, image_tensor, original_class, TARGET_CLASS,
                epsilon, UNTARGETED_ITERATIONS, MAX_ITERATIONS - UNTARGETED_ITERATIONS, device
            )
            results[epsilon]['hybrid'].append((hybrid_iter, hybrid_success))
            status_hybrid = "OK" if hybrid_success else "FAIL"
            print(f"Hybrid (100+targeted): {hybrid_iter:4d} iterations [{status_hybrid}]")

            # Comparaison
            diff = hybrid_iter - direct_iter
            if diff < 0:
                print(f"  -> Hybride GAGNE {-diff} itérations")
            elif diff > 0:
                print(f"  -> Direct GAGNE {diff} itérations")
            else:
                print(f"  -> Égalité")

    # Résumé
    print("\n")
    print("=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)
    print(f"{'Epsilon':<10} {'Méthode':<20} {'Moyenne':<12} {'Succès':<10}")
    print("-" * 70)

    for epsilon in EPSILONS:
        direct_iters = [r[0] for r in results[epsilon]['direct']]
        direct_success = sum(1 for r in results[epsilon]['direct'] if r[1])
        hybrid_iters = [r[0] for r in results[epsilon]['hybrid']]
        hybrid_success = sum(1 for r in results[epsilon]['hybrid'] if r[1])

        avg_direct = np.mean(direct_iters)
        avg_hybrid = np.mean(hybrid_iters)

        print(f"{epsilon:<10} {'Direct':<20} {avg_direct:<12.1f} {direct_success}/{len(SEEDS)}")
        print(f"{'':<10} {'Hybrid (100+target)':<20} {avg_hybrid:<12.1f} {hybrid_success}/{len(SEEDS)}")

        diff = avg_hybrid - avg_direct
        if diff < 0:
            print(f"{'':<10} -> Hybride gagne en moyenne {-diff:.1f} itérations")
        else:
            print(f"{'':<10} -> Direct gagne en moyenne {diff:.1f} itérations")
        print()


if __name__ == "__main__":
    main()
