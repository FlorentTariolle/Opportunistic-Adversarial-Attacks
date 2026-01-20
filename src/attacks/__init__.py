"""Adversarial attack implementations for black-box scenarios."""

from .base import BaseAttack
from .simba import SimBA

__all__ = ['BaseAttack', 'SimBA']
