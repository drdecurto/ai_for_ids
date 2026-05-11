"""
src/models/seedutils.py

Per-model seeding helpers for the multi-seed validation protocol.

The existing model wrappers in ``src.models.trainable.*`` instantiate the
underlying classifier with a hard-coded ``random_state`` value drawn from
``src.models.config``. For the multi-seed protocol described in the
paper (5-seed bootstrap CIs, paired Wilcoxon tests, etc.) we need a fresh
classifier whose seed varies per run. The helpers in this module
construct seeded model instances without mutating the global config.

Two layers of seeding are exposed:

    1. ``set_global_seed(seed)`` — sets the seeds of the standard library
       ``random`` module, NumPy, and (if installed) PyTorch. This ensures
       that any non-explicit randomness inside the model code paths
       behaves reproducibly.

    2. ``build_seeded_model(model_name, seed, **kwargs)`` — returns a
       model wrapper instance whose internal random_state has been
       redirected to the given seed.

The two foundation-model wrappers that hard-code seeds in their config
files (TabPFN/TabPFN 2.5, TabICL, TabSTAR) and the classical Random
Forest baseline are all supported.
"""

from __future__ import annotations

import logging
import os
import random
from copy import deepcopy
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Global seed handling
# ---------------------------------------------------------------------------
def set_global_seed(seed: int) -> None:
    """
    Seed every stochastic library used in the codebase.

    Call this once at the top of each per-seed iteration in the
    multi-seed runner. It is cheap and idempotent.

    Args:
        seed: Integer seed to broadcast to ``random``, NumPy, and PyTorch
            (if importable). Also exported to ``PYTHONHASHSEED`` for any
            child processes spawned during the run.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        # PyTorch is required by all foundation models in the panel,
        # but we don't make this a hard dependency of the helper itself
        # so that the classical-baseline-only path keeps working.
        pass


# ---------------------------------------------------------------------------
# Per-model seeded constructors
# ---------------------------------------------------------------------------
def build_seeded_model(model_name: str, seed: int, **overrides: Any):
    """
    Build a fresh model wrapper instance whose internal random_state is
    set to the requested seed.

    The underlying constructors in ``src.models.trainable.*`` read their
    configuration from ``src.models.config``. To avoid mutating the
    module-level config (which would leak into other code paths), this
    function copies the relevant config dict, overrides ``random_state``,
    monkeypatches the config back for the duration of the constructor
    call, and then restores the original value.

    Args:
        model_name: One of ``"tabpfn"``, ``"tabpfn_2.5"``, ``"tabicl"``,
            ``"tabstar"``, ``"random_forest"``, ``"logistic_regression"``,
            ``"linear_svc"``, ``"decision_tree"``, ``"k_neighbors"``,
            ``"tabnet"``.
        seed: Integer seed for the model.
        **overrides: Optional keyword arguments forwarded to the model
            wrapper constructor (e.g., ``name="tabpfn_seed3"``).

    Returns:
        A fresh model-wrapper instance, ready to ``.fit(...)`` and
        ``.predict(...)``.
    """
    name = model_name.lower().strip()

    if name in ("tabpfn", "tabpfn_2", "tabpfn_2.5"):
        return _build_seeded_tabpfn(name, seed, **overrides)
    if name == "tabicl":
        return _build_seeded_tabicl(seed, **overrides)
    if name == "tabstar":
        return _build_seeded_tabstar(seed, **overrides)
    if name in ("random_forest", "rf"):
        return _build_seeded_random_forest(seed, **overrides)
    if name in ("logistic_regression", "logreg", "lr"):
        return _build_seeded_logistic_regression(seed, **overrides)
    if name in ("linear_svc", "svc", "svm"):
        return _build_seeded_linear_svc(seed, **overrides)
    if name in ("decision_tree", "dt"):
        return _build_seeded_decision_tree(seed, **overrides)
    if name in ("k_neighbors", "knn"):
        return _build_seeded_kneighbors(**overrides)
    if name == "tabnet":
        return _build_seeded_tabnet(seed, **overrides)

    raise ValueError(
        f"Unknown model name '{model_name}'. Supported: "
        "tabpfn, tabpfn_2.5, tabicl, tabstar, random_forest, "
        "logistic_regression, linear_svc, decision_tree, k_neighbors, "
        "tabnet."
    )


# ---------------------------------------------------------------------------
# Internal: TabPFN / TabPFN 2.5
# ---------------------------------------------------------------------------
def _build_seeded_tabpfn(name_alias: str, seed: int, **overrides: Any):
    """
    Construct a seeded TabPFN wrapper.

    TabPFN's wrapper reads three places when picking up a seed:
      - TABPFN_CONFIG['random_state']
      - TABPFN_MANY_CLASS_CONFIG['random_state']
      - The TabPFNClassifier itself (set via
        TabPFNClassifier.create_default_for_version)

    We monkeypatch the config dicts for the duration of the constructor
    call (so the underlying ``TabPFNClassifier`` picks up the seed
    automatically) and restore the originals afterwards.
    """
    from src.models import config as cfg
    from src.models.trainable.tabpfn import TabPFNModel

    version = "2" if name_alias == "tabpfn_2" else (
        "2.5" if name_alias in ("tabpfn", "tabpfn_2.5") else "2.5"
    )
    # Conventional naming: "tabpfn" (no suffix) defaults to the 2.5
    # release because that is the new entrant the paper highlights.

    saved_tabpfn = deepcopy(cfg.TABPFN_CONFIG)
    saved_many = deepcopy(cfg.TABPFN_MANY_CLASS_CONFIG)

    try:
        cfg.TABPFN_CONFIG["random_state"] = int(seed)
        cfg.TABPFN_MANY_CLASS_CONFIG["random_state"] = int(seed)

        model_name = overrides.pop("name", f"tabpfn_v{version}_seed{seed}")
        model = TabPFNModel(name=model_name, version=version, **overrides)
    finally:
        cfg.TABPFN_CONFIG.clear()
        cfg.TABPFN_CONFIG.update(saved_tabpfn)
        cfg.TABPFN_MANY_CLASS_CONFIG.clear()
        cfg.TABPFN_MANY_CLASS_CONFIG.update(saved_many)

    return model


# ---------------------------------------------------------------------------
# Internal: TabICL
# ---------------------------------------------------------------------------
def _build_seeded_tabicl(seed: int, **overrides: Any):
    from src.models import config as cfg
    from src.models.trainable.tabicl import TabICLModel

    saved = deepcopy(cfg.TABICL_CONFIG)
    try:
        cfg.TABICL_CONFIG["random_state"] = int(seed)
        model_name = overrides.pop("name", f"tabicl_seed{seed}")
        model = TabICLModel(name=model_name, **overrides)
    finally:
        cfg.TABICL_CONFIG.clear()
        cfg.TABICL_CONFIG.update(saved)
    return model


# ---------------------------------------------------------------------------
# Internal: TabSTAR
# ---------------------------------------------------------------------------
def _build_seeded_tabstar(seed: int, **overrides: Any):
    from src.models import config as cfg
    from src.models.trainable.tabstar import TabSTARModel

    saved = deepcopy(cfg.TABSTAR_CONFIG)
    try:
        cfg.TABSTAR_CONFIG["random_state"] = int(seed)
        model_name = overrides.pop("name", f"tabstar_seed{seed}")
        model = TabSTARModel(name=model_name, **overrides)
    finally:
        cfg.TABSTAR_CONFIG.clear()
        cfg.TABSTAR_CONFIG.update(saved)
    return model


# ---------------------------------------------------------------------------
# Internal: Classical baselines
# ---------------------------------------------------------------------------
def _build_seeded_random_forest(seed: int, **overrides: Any):
    """
    The Axis-1 Random Forest is heavily pruned (n_estimators=15,
    max_depth=8) for the operational benchmark. We override only its
    random_state.
    """
    from src.models import config as cfg
    from src.models.trainable.ml import MLClassifier
    from sklearn.ensemble import RandomForestClassifier

    rf_cfg = deepcopy(cfg.RANDOM_FOREST_CONFIG)
    rf_cfg["random_state"] = int(seed)
    rf_cfg.update(overrides)
    name = overrides.pop("name", f"random_forest_seed{seed}")
    return MLClassifier(name=name, model=RandomForestClassifier(**rf_cfg))


def _build_seeded_logistic_regression(seed: int, **overrides: Any):
    from src.models import config as cfg
    from src.models.trainable.ml import MLClassifier
    from sklearn.linear_model import LogisticRegression

    lr_cfg = deepcopy(cfg.LOGISTIC_REGRESSION_CONFIG)
    lr_cfg["random_state"] = int(seed)
    lr_cfg.update(overrides)
    name = overrides.pop("name", f"logistic_regression_seed{seed}")
    return MLClassifier(name=name, model=LogisticRegression(**lr_cfg))


def _build_seeded_linear_svc(seed: int, **overrides: Any):
    from src.models import config as cfg
    from src.models.trainable.ml import MLClassifier
    from sklearn.svm import LinearSVC

    svc_cfg = deepcopy(cfg.LINEAR_SVC_CONFIG)
    svc_cfg["random_state"] = int(seed)
    svc_cfg.update(overrides)
    name = overrides.pop("name", f"linear_svc_seed{seed}")
    return MLClassifier(name=name, model=LinearSVC(**svc_cfg))


def _build_seeded_decision_tree(seed: int, **overrides: Any):
    from src.models import config as cfg
    from src.models.trainable.ml import MLClassifier
    from sklearn.tree import DecisionTreeClassifier

    dt_cfg = deepcopy(cfg.DECISION_TREE_CONFIG)
    dt_cfg["random_state"] = int(seed)
    dt_cfg.update(overrides)
    name = overrides.pop("name", f"decision_tree_seed{seed}")
    return MLClassifier(name=name, model=DecisionTreeClassifier(**dt_cfg))


def _build_seeded_kneighbors(**overrides: Any):
    """
    KNN is deterministic w.r.t. fit, so the "seed" only affects any
    upstream randomness (data shuffling, PCA fit). We still expose this
    for API symmetry.
    """
    from src.models import config as cfg
    from src.models.trainable.ml import MLClassifier
    from sklearn.neighbors import KNeighborsClassifier

    knn_cfg = deepcopy(cfg.KNEIGHBORS_CONFIG)
    knn_cfg.update(overrides)
    name = overrides.pop("name", "k_neighbors")
    return MLClassifier(name=name, model=KNeighborsClassifier(**knn_cfg))


def _build_seeded_tabnet(seed: int, **overrides: Any):
    """
    TabNet's wrapper accepts a fixed config; we set both the model's
    random_state and the global PyTorch seed beforehand.
    """
    from src.models.trainable.tabnet import TabNetModel

    set_global_seed(seed)  # TabNet relies on torch's RNG for init
    name = overrides.pop("name", f"tabnet_seed{seed}")
    model = TabNetModel(name=name, **overrides)

    # Best-effort: if the underlying TabNet exposes seed, set it.
    try:
        if hasattr(model, "model") and hasattr(model.model, "seed"):
            model.model.seed = int(seed)
    except Exception as exc:  # pragma: no cover
        logging.warning(f"Could not set TabNet inner seed: {exc}")

    return model
