# AI for Intrusion Detection Systems

A framework and bundled experimental data:

> **Tabular Foundation Models and Few-Shot LLMs for IoT Intrusion Detection**

[![Python](https://img.shields.io/badge/python-3.10+-blue)]()

The repository covers both evaluation axes:

- **Axis 1 — Operational foundation-model benchmark.** Traditional ML baselines,
  TabNet, and four tabular foundation models (TabPFN, TabPFN 2.5, TabICL,
  TabSTAR) evaluated on CIC-IDS2017, N-BaIoT, and CIC-UNSW under four data
  variants per dataset, on the standard 70/30 stratified imbalanced test
  partition. Runs through the `src/` framework and the
  `models_use_example.ipynb` / `multiseed_validation_example.ipynb`
  notebooks.

- **Axis 2 — Controlled few-shot LLM benchmark.** Six open-source LLMs from
  four vendors (Meta Llama 3.1/3.3, Alibaba Qwen3, Nous Hermes-4, Google
  Gemma-3) evaluated on CIC-IDS2017, N-BaIoT, CSE-CIC-IDS2018, and TON-IoT
  under a balanced 50-per-class protocol with 5-seed bootstrap CIs and
  Bonferroni-corrected Mann–Whitney *U* tests against a Random Forest
  anchor. Runs through the standalone `ids_llm_nebius_v7.ipynb` notebook
  against the Nebius OpenAI-compatible endpoint. **Complete results for all
  seven paper configurations (plus one CIC-IDS2017 replicate) are bundled**
  — no re-run is required to inspect any figure or table the paper
  references.

The two axes use the same datasets but **different test-set construction**
and the accuracy numbers from the two protocols are **not directly
comparable**.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Quickstart](#quickstart)
   - [Axis 1 — Operational foundation-model benchmark](#axis-1--operational-foundation-model-benchmark)
   - [Axis 2 — Controlled few-shot LLM benchmark](#axis-2--controlled-few-shot-llm-benchmark)
3. [Reproducibility map: paper figures and tables](#reproducibility-map-paper-figures-and-tables)
4. [Protocol separation: why the two axes are not directly comparable](#protocol-separation-why-the-two-axes-are-not-directly-comparable)
5. [Modules — Specs](#modules--specs)
   - [DATA](#data)
   - [MODELS](#models)
   - [PIPELINES](#pipelines)
6. [Citation](#citation)

## Project Structure

```plaintext
ai_for_ids/
├── README.md                              # this file
├── requirements.txt                       # pinned env, Python 3.10–3.12
│
├── src/                                   # Axis 1 framework (importable package)
│   ├── data/
│   │   ├── tabular/                       # CIC-IDS2017, N-BaIoT, CIC-UNSW loaders
│   │   ├── config.py
│   │   ├── data_augmentation.py
│   │   └── utils.py
│   ├── models/
│   │   ├── API/llm.py                     # legacy Mistral / Gemini wrappers
│   │   ├── trainable/                     # ml.py, tabnet, tabpfn, tabicl,
│   │   │                                  # tabstar, tabdpt
│   │   ├── config.py
│   │   ├── seedutils.py                   # build_seeded_model(name, seed)
│   │   └── utils.py
│   └── pipelines/
│       ├── pipelines.py                   # TTPipeline, TAPipeline
│       ├── multiseed.py                   # MultiSeedRunner, run_panel
│       └── utils.py
│
├── models_use_example.ipynb               # Axis 1 — single-seed walkthrough
├── multiseed_validation_example.ipynb     # Axis 1 — multi-seed with bootstrap CIs
├── llms_use_example.ipynb                 # legacy: Mistral / Gemini API demos
│                                          # (superseded by ids_llm_nebius_v7
│                                          #  for the paper's LLM evaluation)
│
├── ids_llm_nebius_v7.ipynb                # Axis 2 — standalone, self-contained
│                                          # Colab notebook (Nebius endpoint)
│
└── data_llm/                              # Axis 2 — bundled results, eight runs.
    │                                      # All seven paper configurations are
    │                                      # included plus a CIC-IDS2017 replicate
    │                                      # held in reserve as a sanity check.
    │                                      # Every directory has the same internal
    │                                      # structure (one subfolder per E1–E9 /
    │                                      # R1 / R2 experiment, plus the master
    │                                      # workbook). See the run map below.
    │
    ├── ids_llm_v7/                        # CIC-IDS2017 grouped (9 cls)   — primary
    │                                      #   source of all paper figures (Fig 4–9)
    ├── ids_llm_v7_2/                      # CIC-IDS2017 fine (15 cls)
    ├── ids_llm_grouped_v7_3/              # N-BaIoT grouped (3 cls)
    ├── ids_llm_v7_3/                      # N-BaIoT fine (11 cls)
    ├── ids_llm_grouped_v7_5/              # CSE-CIC-IDS2018 grouped (7 cls)
    ├── ids_llm_v7_5/                      # CSE-CIC-IDS2018 fine (15 cls)
    ├── ids_llm_v7_6/                      # TON-IoT (10 cls)
    └── ids_llm_v7_7/                      # CIC-IDS2017 grouped REPLICATE
                                           #   (sanity check, not in paper)

    Per-directory contents (identical layout in each):
        r1_multiseed/                      # R1 cross-vendor multi-seed
        r2_perclass_multiseed/             # R2 per-class delta vs RF
        e1_cross_model/   e2_kshot/        # E1 single-seed; E2 k-shot scaling
        e3_format/        e4_selfcons/     # E3 prompt-format; E4 n_vote sweep
        e5_cost/          e6_perclass/     # E5 USD / sec; E6 confusion + F1
        e7_convergence/   e8_hybrid/       # E7 N_test sweep; E8 LLM+RF hybrid
        e9_scaling/                        # E9 within-family scaling
        ids_llm_v7_master.xlsx             # consolidated workbook
```

## Quickstart

### Environment

The repo targets Python 3.10–3.12. The pinned environment is in
`requirements.txt`:

```bash
python -m venv .venv && source .venv/bin/activate     # or your preferred env tool
pip install -r requirements.txt
```

GPU is optional for Axis 1 (CPU works for the prior-fitted models; TabSTAR's
LoRA fine-tune benefits from a single H200 or comparable). Axis 2 only needs
internet access to the Nebius endpoint and a CPU; no local GPU required.

### Secrets

Four credentials are read from environment variables (use whatever secret
manager you prefer; in Colab use `userdata.get(...)`):

| Variable           | Used by        | Notes                                                |
|--------------------|----------------|------------------------------------------------------|
| `KAGGLE_USERNAME`  | Axis 1, Axis 2 | Required for the Kaggle dataset mirrors.             |
| `KAGGLE_KEY`       | Axis 1, Axis 2 | Same — generate one in your Kaggle account settings. |
| `HF_TOKEN`         | Axis 1         | Required for TabPFN 2.5 (gated on Hugging Face).     |
| `NEBIUS_API_KEY`   | Axis 2         | Inference endpoint for the open-source LLMs.         |

### Axis 1 — Operational foundation-model benchmark

The framework follows a *module → pipeline → evaluation* pattern: pick a
dataset module, pick a model module, run a pipeline. Single-seed end-to-end
example, drawing from `models_use_example.ipynb`:

```python
from src.data import CICIDS2017
from src.models.trainable import TabPFNModel, TabICLModel
from src.pipelines import TTPipeline

dataset = CICIDS2017(classes_mapping=False, pca=False)   # V1
dataset.load()

for ModelClass in (TabPFNModel, TabICLModel):
    pipeline = TTPipeline(dataset=dataset, model=ModelClass())
    pipeline.run()
    print(pipeline.results())
```

For the multi-seed protocol (the R1 equivalent on Axis 1, flagged in the
paper's Limitations as a follow-up direction), see
`multiseed_validation_example.ipynb`. The relevant entry points in code:

```python
from src.pipelines.multiseed import MultiSeedRunner, run_panel
```

`MultiSeedRunner` re-runs the pipeline across N seeds and reports seed-mean
accuracy / macro-F1 with bootstrap 95 % CIs. `run_panel(...)` sweeps a whole
model panel against shared seeds, with paired Wilcoxon p-values for
cell-by-cell comparisons. Recommended seed budget is 5 (matches Axis 2);
three is a faster floor; two gives no meaningful CI.

### Axis 2 — Controlled few-shot LLM benchmark

The Axis 2 study is a **single self-contained notebook**,
`ids_llm_nebius_v7.ipynb`, that runs against the Nebius OpenAI-compatible
endpoint. It does its own data loading, preprocessing, feature preselection,
and ICL / test split, so it does not depend on the `src/` package.

**You do not need to run the notebook to inspect any paper artifact.** All
seven paper configurations are already bundled under `data_llm/ids_llm_v7*/` (see
the run map below); every figure, table, and CSV the paper references is
directly readable from disk. Run the notebook only if you want to extend
the benchmark, change the protocol, swap in a new vendor, or re-validate
the numbers from scratch.

#### Run map: directory → configuration

The notebook is parameterised by two switches that determine which
configuration is being run:

```python
DATASET        = "cic-ids2017"     # cic-ids2017 | unsw-nb15 | n-baiot |
                                   # cse-cic-ids2018 | ton-iot
CLASS_GROUPING = "grouped"         # fine | grouped
```

The eight bundled run directories were each produced by one combination
of these switches. The paper covers seven configurations; the eighth
directory is a CIC-IDS2017 replicate held in reserve for the sanity check
that R1's accuracy parity reproduces under a second seed sweep:

| Directory                  | `DATASET`          | `CLASS_GROUPING` | Classes | In paper? |
|----------------------------|--------------------|------------------|---------|-----------|
| `data_llm/ids_llm_v7/`              | `cic-ids2017`      | `grouped`        | 9       | ✓ primary |
| `data_llm/ids_llm_v7_2/`            | `cic-ids2017`      | `fine`           | 15      | ✓         |
| `data_llm/ids_llm_grouped_v7_3/`    | `n-baiot`          | `grouped`        | 3       | ✓         |
| `data_llm/ids_llm_v7_3/`            | `n-baiot`          | `fine`           | 11      | ✓         |
| `data_llm/ids_llm_grouped_v7_5/`    | `cse-cic-ids2018`  | `grouped`        | 7       | ✓         |
| `data_llm/ids_llm_v7_5/`            | `cse-cic-ids2018`  | `fine`           | 15      | ✓         |
| `data_llm/ids_llm_v7_6/`            | `ton-iot`          | `fine`           | 10      | ✓         |
| `data_llm/ids_llm_v7_7/`            | `cic-ids2017`      | `grouped`        | 9       | replicate (not used in paper) |

`data_llm/ids_llm_v7/` is the primary directory: every figure that the paper
renders (Fig. 4–9) is drawn from this run. The other six paper-covered
runs feed only the R1 numbers in Table 2; their per-configuration
figures exist in their own directories but are held in reserve in case
a reviewer asks to see them.

#### SCALE knob (only relevant if you re-run the notebook)

The notebook has a single `SCALE` knob near the top:

```python
SCALE = "paper"   # 'quick' | 'paper'
```

- `quick` — `N_TEST = 8`, R1 multi-seed off, no convergence sweep. ~5 min,
  ~$0.50 of Nebius credit. Use this to verify the pipeline runs end-to-end
  before committing to a full run.
- `paper` — `N_TEST = 50`, R1 with 5 seeds, full E7 convergence grid
  `{16, 32, 64, 128}`. ~30–90 min wall-clock, ~$20–40 of Nebius credit per
  configuration. **Every bundled directory was produced under
  `SCALE = "paper"`.**

The notebook checkpoints per-seed CSVs after every seed, so a transient
runtime disconnect or rate-limit pause does not lose progress. On
completion it emits a master Excel workbook (`ids_llm_v7_master.xlsx`)
with one sheet per experiment.


### Table 2 — R1 multi-seed accuracy + CIs (7 rows, 7 directories)

| Paper row                                | File                                                          |
|------------------------------------------|---------------------------------------------------------------|
| CIC-IDS2017 grouped (9 cls)              | `data_llm/ids_llm_v7/r1_multiseed/r1_ci.csv`                           |
| CIC-IDS2017 fine    (15 cls)             | `data_llm/ids_llm_v7_2/r1_multiseed/r1_ci.csv`                         |
| N-BaIoT      grouped (3 cls)             | `data_llm/ids_llm_grouped_v7_3/r1_multiseed/r1_ci.csv`                 |
| N-BaIoT      fine    (11 cls)            | `data_llm/ids_llm_v7_3/r1_multiseed/r1_ci.csv`                         |
| CSE-CIC-IDS2018 grouped (7 cls)          | `data_llm/ids_llm_grouped_v7_5/r1_multiseed/r1_ci.csv`                 |
| CSE-CIC-IDS2018 fine    (15 cls)         | `data_llm/ids_llm_v7_5/r1_multiseed/r1_ci.csv`                         |
| TON-IoT      (10 cls)                    | `data_llm/ids_llm_v7_6/r1_multiseed/r1_ci.csv`                         |
| *Replicate (not in paper, sanity check)* | `data_llm/ids_llm_v7_7/r1_multiseed/r1_ci.csv`                         |

The Mann–Whitney *U* p-values reported in Table 2 are in the matching
`r1_mwu.csv` of each directory.

### Table 3 — R2 per-class F1 deltas (per-configuration)

Table 3 for CIC-IDS2017 grouped only; the same analysis is
available for every configuration:

| Paper row (CIC-IDS2017 grouped)          | File                                                          |
|------------------------------------------|---------------------------------------------------------------|
| Table 3 — bootstrap CIs                  | `data_llm/ids_llm_v7/r2_perclass_multiseed/r2_perclass_ci.csv`         |
| Table 3 — typeset journal version        | `data_llm/ids_llm_v7/r2_perclass_multiseed/r2_journal_table.tex`       |

Equivalent files exist under `data_llm/ids_llm_v7_2/r2_perclass_multiseed/`,
`data_llm/ids_llm_grouped_v7_3/r2_perclass_multiseed/`, etc., for the other six
configurations.

### Other tables (single-configuration, from `data_llm/ids_llm_v7/`)

| Paper element                                | File                                                          |
|----------------------------------------------|---------------------------------------------------------------|
| Table 4 — E9 within-family scaling pairs     | `data_llm/ids_llm_v7/e9_scaling/e9_pairs.csv`                          |
| Table 5 — E5 cost / latency / tokens         | `data_llm/ids_llm_v7/e5_cost/e5_cost.csv`                              |
| Table 6 — E8 hybrid LLM+RF accuracy + CIs    | `data_llm/ids_llm_v7/e8_hybrid/e8_hybrid_ci.csv`                       |

### Figures (all from `data_llm/ids_llm_v7/`, the CIC-IDS2017 grouped primary run)

| Paper element                                | File                                                          |
|----------------------------------------------|---------------------------------------------------------------|
| Fig. 4 — R1 multi-seed CI bars               | `data_llm/ids_llm_v7/r1_multiseed/r1_ci_bars.{pdf,png}`                |
| Fig. 5 — R2 per-class delta                  | `data_llm/ids_llm_v7/r2_perclass_multiseed/r2_perclass_delta.{pdf,png}`|
| Fig. 6 — E6 confusion matrix + per-class F1  | `data_llm/ids_llm_v7/e6_perclass/e6_perclass.{pdf,png}`                |
| Fig. 7 — E9 within-family scaling curve      | `data_llm/ids_llm_v7/e9_scaling/e9_scaling_curve.{pdf,png}`            |
| Fig. 8 — E5 cost–accuracy Pareto             | `data_llm/ids_llm_v7/e5_cost/e5_pareto.{pdf,png}`                      |
| Fig. 9 — E8 hybrid combination accuracy      | `data_llm/ids_llm_v7/e8_hybrid/e8_hybrid.{pdf,png}`                    |

Per-configuration versions of every figure are available in each run
directory under the same filename (e.g.
`data_llm/ids_llm_v7_2/r1_multiseed/r1_ci_bars.pdf` for the CIC-IDS2017 fine version
of Fig. 4); the paper renders only the primary one for space.

### Diagnostic experiments (not in the main paper, available for review)

| Element                                | File (per directory)                                           |
|----------------------------------------|----------------------------------------------------------------|
| E1 cross-family LLM sweep              | `e1_cross_model/e1_summary.csv` + `e1_bars.{pdf,png}`          |
| E2 *k*-shot scaling                    | `e2_kshot/e2_summary.csv` + `e2_kshot.{pdf,png}`               |
| E3 prompt-format ablation              | `e3_format/e3_summary.csv` + `e3_format.{pdf,png}`             |
| E4 self-consistency sweep              | `e4_selfcons/e4_summary.csv` + `e4_selfcons.{pdf,png}`         |
| E7 sample-size convergence             | `e7_convergence/e7_convergence.csv` + `.{pdf,png}`             |
| Consolidated workbook                  | `ids_llm_v7_master.xlsx` (one sheet per experiment)            |

These files appear inside every `data_llm/ids_llm_v7*/` directory; replace the
prefix with the directory of interest to inspect a specific configuration.


## Protocol separation: why the two axes are not directly comparable

The Axis 1 test partition is the full 30 % stratified random sample of the
cleaned dataset, which preserves the **operational class distribution**.
On CIC-IDS2017 this means BENIGN dominates at ~80 %; Heartbleed appears at
~10⁻⁵. A model that always predicts the majority class scores ~0.80 on the
Axis 1 protocol.

The Axis 2 test split is balanced at exactly **50 rows per class**. A model
that always predicts the majority class scores ~1 / *C* where *C* is the
number of classes — 0.11 on N-BaIoT fine, 0.07 on CIC-IDS2017 fine.

The two protocols answer different questions:

- *Axis 1* — how well does the model perform on data drawn from the
  realistic deployment distribution?
- *Axis 2* — how well does the model discriminate between classes when
  forced to attend to all of them equally, including rare ones?

The two accuracy numbers therefore live on different axes. Reading them
side by side as if they were comparable produces systematically misleading
conclusions about LLM viability for IDS.

---

# Modules — Specs

The remaining sections document the `src/` framework (Axis 1). The
specification of how to use each module is contained in the code; this
section indexes the configurable parameters. The Axis 2 notebook
(`ids_llm_nebius_v7.ipynb`) does not depend on these modules — it is
self-contained — and is documented inline.

## DATA

This category wraps all the modules related to the datasets. Each dataset
has its own module, which handles download, preprocessing, and adaptation.

### Modules hierarchy

```
BaseDataset
├── TabularDataset
│   ├── CIC-IDS2017
│   ├── N-BaIoT
│   └── CIC-UNSW
└── (other possible types)
```

### Configurations

#### Kaggle configuration

- `KAGGLE_USERNAME`: Kaggle username
- `KAGGLE_KEY`:  Kaggle key
    - ⚠️  Even if these are represented as constants, Kaggle authentication
      requires them to be set as **environment variables**.
    - ⚠️  Kaggle allows other authentication methods, but the environment-
      variable approach is the most convenient. When loading a dataset
      imported from Kaggle (e.g. N-BaIoT) an error is raised if authentication
      is not properly set.

#### CIC-IDS2017 configuration

- `CIC_IDS2017_CLASSES_MAPPING`: Mapping of classes for CIC-IDS2017.

- `CICIDS2017_DEFAULT_CONFIG`
    - **pca**: Whether to use PCA by default.
    - **classes_mapping**: Whether to use class mapping by default.

#### N-BaIoT configuration

- `N_BAIOT_DEFAULT_CONFIG`
    - **64_to_32_quantization**: Whether to use 64-bit to 32-bit quantization by default.
    - **pca**: Whether to use PCA by default.
    - **classes_mapping**: Whether to use class mapping by default.

- `N_BAIOT_CLASSES_MAPPING`: Mapping of classes for N-BaIoT.

#### CIC-UNSW configuration

- `CICUNSW_DEFAULT_CONFIG`
    - **pca**: Whether to use PCA by default.

#### Data augmentation

- `SMOTE_CONFIG`
    - **class_samples_threshold**: Percentage that indicates whether to augment a class samples or not depending on the class with the most number of samples.
    - **n**: The amount of times to augment the data.
    - **alpha**: Alpha for the beta distribution.
    - **beta**: Beta for the beta distribution.

- `TABPFN_DATA_GENERATOR_CONFIG`
    - **class_samples_threshold**: Percentage that multiplies the max amount of samples in a class to determine the classes being augmented.
    - **t**: Temperature parameter for sampling. Higher values produce more diverse samples, lower values produce more deterministic samples.
    - **n_permutations**: Number of feature permutations to use for generation. More permutations may provide more robust results but increase computation time.

- `TABPFN_DATA_GENERATOR_MODEL_CONFIG` — see [TabPFN](#tabpfn-configuration-parameters).
- `TABPFN_DATA_GENERATOR_EXPERT_MODEL_CONFIG` — see [TabPFN](#tabpfn-configuration-parameters).

## MODELS

The models are divided into LLMs (API-based) and trainable models (ML or DL).
For further explanation of the behaviour or construction of these models,
look into the example notebooks or the in-source docstrings.

### Modules hierarchy

```
BaseModel
├── TrainableModel
│   ├── SklearnTrainableModel
│   │   ├── MLClassifier
│   │   │   ├── PreConfigured_DecisionTree
│   │   │   ├── PreConfigured_KNeighbors
│   │   │   ├── PreConfigured_LinearSVC
│   │   │   ├── PreConfigured_LogisticRegression
│   │   │   └── PreConfigured_RandomForest
│   │   ├── TabNetModel
│   │   ├── TabPFNModel
│   │   ├── TabICLModel
│   │   ├── TabSTARModel
│   │   └── TabDPTModel
│   └── (other kinds of models, like torch-based DL models)
└── LLModel
    ├── Mistral
    └── Gemini
```

> **Note on the LLM API wrappers.** `src/models/API/llm.py` provides the
> Mistral and Gemini wrappers used in the **prior conference paper**'s
> preliminary LLM evaluation (single vendor, 100 samples, no statistical
> validation). They are kept here for backward compatibility with
> `llms_use_example.ipynb`. The **paper's Axis 2 LLM evaluation** uses the
> Nebius endpoint and lives entirely in `ids_llm_nebius_v7.ipynb`; it does
> not go through this wrapper.

### Configurations

Some models can have their configuration or training modified via
constructor arguments. Where finer control is needed, the configuration
file is the source of truth.

#### Hugging Face configuration

- `HF_TOKEN`: Hugging Face token.
    - ⚠️  Needed for models whose weights load from Hugging Face and are
      not fully open (e.g. TabPFN 2.5). Set as an environment variable, or
      in the configuration file (not recommended).

#### ML models configuration parameters

- `LOGISTIC_REGRESSION_CONFIG`
    - **multi_class**: `{'auto', 'ovr', 'multinomial'}`
    - **max_iter**: Maximum number of iterations for the solver to converge.
    - **solver**: Algorithm to use in the optimization problem.
    - **C**: Inverse of regularization strength; must be a positive float.
    - **random_state**: Used when `solver == 'sag', 'saga' or 'liblinear'` to shuffle the data.

- `LINEAR_SVC_CONFIG`
    - **C**: Regularization parameter. Strength is inversely proportional to *C*.
    - **random_state**: Pseudo-random number generation for shuffling the data.
    - **tol**: Tolerance for stopping criteria.
    - **max_iter**: Maximum number of iterations to be run.

- `RANDOM_FOREST_CONFIG`
    - **n_estimators**: The number of trees in the forest.
    - **max_depth**: The maximum depth of the tree.
    - **max_features**: The number of features to consider when looking for the best split.
    - **random_state**: Controls the bootstrap and feature-sampling randomness.

- `KNEIGHBORS_CONFIG`
    - **n_neighbors**: Number of neighbors to use.

- `DECISION_TREE_CONFIG`
    - **max_depth**: The maximum depth of the tree.

#### TabNet configuration parameters

- `TABNET_PRETRAINER_CONFIG`
    - **n_d**, **n_a**: Dimensions of the prediction / attention layers.
    - **n_steps**: Number of successive steps in the network.
    - **gamma**: Scaling factor for attention updates (typically 1.0–2.0).
    - **n_independent**, **n_shared**: Number of GLU layers in each block.
    - **lambda_sparse**: Sparsity loss parameter.
    - **optimizer_fn**, **optimizer_params**: Optimizer and its parameters.

- `TABNET_PRETRAINING_PARAMS`
    - **max_epochs**: Maximum pretraining epochs.
    - **batch_size**: Batch size.
        - ⚠️  If the batch size exceeds the number of provided data samples (training or validation) an error is raised.
    - **patience**: Number of epochs with no improvement after which training stops.

- `TABNET_CONFIG`
    - **n_d**, **n_a**, **n_steps**, **gamma**, **n_independent**, **n_shared**, **lambda_sparse**: As above.
    - **optimizer_fn**, **optimizer_params**, **scheduler_fn**, **scheduler_params**: Optimizer / scheduler setup.
    - **mask_type**: Masking function.

- `TABNET_TRAINING_PARAMS`
    - **eval_metric**, **max_epochs**, **batch_size**, **patience**: Standard training controls.

#### TabPFN configuration parameters

- `TABPFN_CONFIG`
    - **n_estimators**: Number of estimators in the TabPFN ensemble.
    - **balance_probabilities**: Whether to balance the probabilities based on the class distribution.
    - **average_before_softmax**: Whether to average estimator predictions before softmax.
    - **device**: Inference device.
    - **ignore_pretraining_limits**: Whether to ignore the model's pre-training limits.
    - **inference_precision**: Inference precision.
    - **fit_mode**: How the data is preprocessed and cached for inference.
    - **memory_saving_mode**: Enable GPU/CPU memory saving mode.
    - **random_state**: Controls model randomness.
    - **n_jobs**: Workers for parallelizable tasks.

- `TABPFN_EXPERT_CONFIG`
    - **CLASS_SHIFT_METHOD**, **FEATURE_SHIFT_METHOD**: Shift methods used during preprocessing for ensembling.
    - **FINGERPRINT_FEATURE**: Whether to add a fingerprint feature.
    - **MAX_NUMBER_OF_CLASSES**, **MAX_NUMBER_OF_FEATURES**, **MAX_NUMBER_OF_SAMPLES**: Pretraining caps.
    - **MAX_UNIQUE_FOR_CATEGORICAL_FEATURES**, **MIN_UNIQUE_FOR_NUMERICAL_FEATURES**: Cardinality thresholds.
    - **OUTLIER_REMOVAL_STD**: SD threshold for outlier removal.
    - **POLYNOMIAL_FEATURES**: Number of degree-2 polynomial features to add.
    - **SUBSAMPLE_SAMPLES**: Subsample size before TabPFN forward pass.
    - **USE_SKLEARN_16_DECIMAL_PRECISION**: Whether to round probabilities to float16 precision.

- `TABPFN_PARAMS`
    - **predicting_batch_size**: Batch size for prediction. Use `-1` for a single batch.

- `TABPFN_MANY_CLASS_CONFIG`
    - **alphabet_size**: Maximum number of classes the base estimator can handle. Must be ≤ `MAX_NUMBER_OF_CLASSES`.
    - **n_estimators**: Number of base estimators (`None` for automatic detection).
    - **n_estimators_redundancy**: Redundancy factor for the auto-calculated number of estimators.
    - **random_state**: Controls codebook randomization.

#### TabICL configuration parameters

- `TABICL_CONFIG`
    - **n_estimators**: Number of ensemble members.
    - **norm_methods**: Normalization methods to try.
    - **feat_shuffle_method**: Feature permutation strategy.
    - **class_shift**: Whether to apply cyclic shifts to class labels.
    - **outlier_threshold**: Z-score threshold for outlier detection and clipping.
    - **softmax_temperature**: Controls prediction confidence.
    - **average_logits**: Whether ensemble averaging is done on logits or probabilities.
    - **use_hierarchical**: Enable hierarchical classification for many-class datasets.
    - **batch_size**: Ensemble batch size (reduce to save RAM).
    - **use_amp**: Automatic mixed precision for faster inference.
    - **model_path**: Where the model checkpoint is stored.
    - **allow_auto_download**: Whether automatic download to the specified path is allowed.
    - **checkpoint_version**: Version of pretrained checkpoint to use.
    - **device**, **random_state**, **n_jobs**, **verbose**: Standard runtime controls.

- `TABICL_PARAMS`
    - **predicting_batch_size**: Batch size for prediction. Use `-1` for a single batch.

#### TabSTAR configuration parameters

- `TABSTAR_CONFIG`
    - **lora_lr**: Learning rate for updating the LoRA adapter.
    - **lora_r**: Rank of the LoRA adapter.
    - **lora_batch**: Batch size when training only with LoRA parameters.
    - **global_batch**: Effective batch size across all devices used in fine-tuning.
    - **max_epochs**: Maximum number of epochs to run for fine-tuning.
    - **patience**: Max epochs without validation improvement before stopping early.
    - **verbose**, **device**, **random_state**, **debug**: Standard runtime controls.

- `TABSTAR_PARAMS`
    - **predicting_batch_size**: Batch size for prediction. Use `-1` for a single batch.

#### TabDPT configuration parameters

- `TABDPT_CONFIG`
    - **inf_batch_size**: Size of inference batches for retrieval-based inference.
    - **normalizer**: Feature normalization (`standard | minmax | robust | power | quantile-uniform | quantile-normal | log1p | None`).
    - **missing_indicators**: Whether to add missing-value indicator features.
    - **clip_sigma**: Clipping threshold for extreme values.
    - **feature_reduction**: Method for reducing feature count (`pca | subsample`).
    - **faiss_metric**: FAISS nearest-neighbour metric (`l2 | ip`).
    - **device**: Inference device.
    - **use_flash**: Whether to use optimized attention kernels.
    - **compile**: Whether to compile the model for improved efficiency.
    - **model_weight_path**: Path to the pre-trained model weights.

#### LLMs configuration parameters (legacy API wrappers)

These cover the Mistral and Gemini wrappers in `src/models/API/llm.py`, used
by the prior conference paper's preliminary LLM evaluation. The paper's
Axis 2 evaluation does **not** use these — it uses
`ids_llm_nebius_v7.ipynb` against the Nebius endpoint.

- `MISTRAL_API_KEY`: Mistral API key.
- `GEMINI_API_KEY`: Gemini API key.
- `BACKOFF_MAX_TRIES`: Maximum number of attempts before giving up.
- `BACKOFF_FACTOR`: The amount by which the backoff time increases on each successive backoff.
- `GEMINI_GENERATION_CONFIG`
    - **top_p**, **temperature**, **candidate_count**: Standard sampling controls.
- `MISTRAL_GENERATION_CONFIG`
    - **top_p**, **temperature**, **n**: Standard sampling controls. We generally recommend altering either `temperature` or `top_p` but not both.

## PIPELINES

### Modules hierarchy

```
BasePipeline
├── TTPipeline
└── TAPipeline

MultiSeedRunner          (src/pipelines/multiseed.py)
SeedResult               (per-seed dataclass)
compare_runners(a, b)    (paired Wilcoxon between two runners)
run_panel(...)           (sweep one model panel on shared seeds)
```

### Multi-seed validation

`src/pipelines/multiseed.py` adds a thin orchestration layer that re-runs the
operational foundation-model benchmark across N seeds, with bootstrap 95 %
confidence intervals on seed-mean accuracy / macro-F1 and paired Wilcoxon
signed-rank tests for cell-by-cell comparisons (e.g. TabPFN vs TabPFN 2.5).
Two prerequisite changes from the upstream single-seed pipeline:

1. **Reproducible splits.** `TabularDataset.train_test_split` now accepts a
   `random_state` argument (default `None`, backward-compatible) and an
   optional `stratify_col` for stratified sampling — recommended for the
   imbalanced IDS datasets so rare classes do not vanish from the test split.

2. **Per-run model seeding.** `src/models/seedutils.py` exposes
   `build_seeded_model(name, seed, **overrides)`, which constructs a fresh
   wrapper instance whose internal `random_state` is set to the requested
   seed. Supported names: `tabpfn`, `tabpfn_2.5`, `tabicl`, `tabstar`,
   `random_forest`, `logistic_regression`, `linear_svc`, `decision_tree`,
   `k_neighbors`, `tabnet`. Module-level configs are not mutated.

The notebook `multiseed_validation_example.ipynb` walks through:

- a single *(dataset, variant)* cell — TabPFN vs TabPFN 2.5 with paired
  Wilcoxon *p*-value;
- the full operational panel via `run_panel(...)`;
- per-seed CSVs and aggregate JSON files saved to disk for downstream
  analysis.

Recommended seed budget is 5 (matches the LLM benchmark protocol used in
Axis 2). Three seeds is a faster floor with wider CIs; two seeds is enough
for sanity-checking but does not give a meaningful CI. TabSTAR is the
expensive model in the panel because of the LoRA fine-tune; running it with
a smaller seed budget (e.g. `seeds=(0, 1)`) while the rest of the panel
runs at five seeds is a reasonable compromise.
