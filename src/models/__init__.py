"""Minimal model package: only the TabPFN wrapper used by the sensitivity run."""
from .trainable.tabpfn import TabPFNModel

__all__ = ["TabPFNModel"]
