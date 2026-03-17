# UROPS: Multimodal In-Hospital Mortality Prediction with Adversarial Regularisation

A deep learning framework for predicting in-hospital patient mortality using the [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) clinical database. The model fuses seven heterogeneous data modalities — vitals, labs, clinical notes, procedures, microbiology, fluid inputs/outputs, and patient demographics — through modality-specific Transformer encoders, and optionally applies adversarial (discriminator) regularisation controlled by a `beta` hyperparameter.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Data](#data)
- [Setup](#setup)
- [Data Preprocessing](#data-preprocessing)
- [Training](#training)
- [Results](#results)
- [Dependencies](#dependencies)

---

## Overview

This project was developed as part of an Undergraduate Research Opportunities Programme (UROPS). The goal is to predict whether a patient will die during their hospital stay using rich, multimodal EHR data from MIMIC-III.

Key design choices:
- **Modality-specific encoders**: each clinical data type (inputs, outputs, labs, notes, CPT procedures, microbiology, demographics) is encoded by a dedicated sub-network.
- **Attention-based fusion**: a learned attention mechanism (`AttFusion`) pools variable-length sequences into fixed-size representations before concatenation.
- **Adversarial regularisation**: a discriminator head is trained alongside the classifier, with its loss weighted by `beta`. This encourages the learned representation to be more robust and less overfit to spurious correlations.
- **Hyperparameter search**: training is orchestrated with [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) and the ASHA scheduler.

---

## Architecture

```
Patient EHR
    │
    ├── Inputs (vitals/fluids)   ──► TransformerEncoderCTE  ─┐
    ├── Outputs (fluid outputs)  ──► TransformerEncoderCTE  ─┤
    ├── Labs                     ──► TransformerEncoderCTE  ─┤
    ├── Microbiology             ──► TransformerEncoderQuad ─┤──► Concat ──► Condensor ──► Classifier ──► Mortality
    ├── Clinical Notes           ──► TransformerEncoder     ─┤                                └──► Discriminator
    ├── CPT Procedures           ──► TransformerEncoder     ─┤
    └── Demographics             ──► Linear                 ─┘
```

### Sub-modules ([`Model-Code/subModules.py`](Model-Code/subModules.py))

| Module | Purpose |
|---|---|
| [`Time2VecPos`](Model-Code/subModules.py) | Learnable sinusoidal time positional encoding |
| [`InitTriplet`](Model-Code/subModules.py) | Embeds `(variable, time, value)` triplets for chart/lab/IO events |
| [`LabQuad`](Model-Code/subModules.py) | Embeds `(specimen, time, organism, interpretation)` quadruplets for microbiology |
| [`TransformerEncoderUnit`](Model-Code/subModules.py) | Single Transformer encoder block with multi-head attention, FFN, layer norm, and dropout |
| [`AttFusion`](Model-Code/subModules.py) | Attention-weighted pooling to collapse a sequence into a single vector |
| [`TransformerEncoderCTE`](Model-Code/subModules.py) | Full encoder for triplet-type modalities (inputs, outputs, labs) |
| [`TransformerEncoderQuad`](Model-Code/subModules.py) | Full encoder for quadruplet-type modalities (microbiology) |
| [`TransformerEncoder`](Model-Code/subModules.py) | Full encoder for text/procedure modalities (notes, CPT) |
| [`MLP`](Model-Code/subModules.py) | 4-layer MLP with GELU activations and dropout; used for both classifier and discriminator heads |
| [`GradientReversal`](Model-Code/gradient_reversal/module.py) | Gradient reversal layer for domain-adversarial training |

### Full Model ([`Model-Code/fullModel.py`](Model-Code/fullModel.py))

[`FullModel`](Model-Code/fullModel.py) is a [`pytorch_lightning.LightningModule`](Model-Code/fullModel.py) that:
1. Encodes each modality independently.
2. Concatenates all representations.
3. Passes through a `condensor` linear layer to produce a final patient embedding.
4. Feeds the embedding to a `classifier` (binary cross-entropy, mortality prediction) and a `discriminator` (adversarial regularisation).
5. Optimises the combined loss: `loss = class_loss + beta * disc_loss`.

---

## Project Structure

```
urops-code/
├── Data-Preprocessing/
│   ├── admission_cats.py       # Saves unique categorical values for demographic encoding
│   ├── data_helperFxns.py      # Date normalisation and per-patient CSV splitting utilities
│   ├── get_SubDOB.py           # Builds subject-ID → date-of-birth mapping and patient list
│   ├── split_Data.py           # Main preprocessing script: processes all MIMIC-III tables
│   └── data-process.txt        # PBS HPC job script for running preprocessing on a GPU cluster
│
├── Model-Code/
│   ├── subModules.py           # All neural network building blocks
│   ├── fullModel.py            # Full multimodal model (LightningModule)
│   ├── patient_Dataset.py      # PyTorch Dataset and data transformation pipeline
│   ├── trainer.py              # Ray Tune training loop and hyperparameter search
│   ├── test_data.py            # Inference / result generation script
│   └── gradient_reversal/
│       ├── __init__.py
│       ├── functional.py       # Custom autograd function for gradient reversal
│       └── module.py           # GradientReversal nn.Module wrapper
│
├── Data-Analysis/
│   ├── Pre-Model Analysis.ipynb   # Exploratory data analysis before modelling
│   └── Post- ML Analysis.ipynb    # Evaluation and analysis of model results
│
├── Results/
│   ├── beta0.csv               # Embeddings, predictions, and labels (beta = 0, no adversarial)
│   ├── beta2.csv               # Embeddings, predictions, and labels (beta = 2)
│   └── beta20.csv              # Embeddings, predictions, and labels (beta = 20)
│
├── images/
│   ├── grl.png                 # Gradient reversal layer diagram
│   └── result.png              # Results visualisation
│
├── requirements.txt
└── README.md
```

---

## Data

This project uses the **MIMIC-III Clinical Database** (v1.4). Access requires credentialed registration on [PhysioNet](https://physionet.org/content/mimiciii/1.4/).

The following MIMIC-III tables are used:

| Table | Modality |
|---|---|
| `PATIENTS.csv` | Demographics (gender, date of birth, mortality flag) |
| `ADMISSIONS.csv` | Demographics (ethnicity, language, religion, insurance, marital status) |
| `LABEVENTS.csv` | Lab results |
| `MICROBIOLOGYEVENTS.csv` | Microbiology cultures and sensitivities |
| `CPTEVENTS.csv` | CPT procedure codes |
| `INPUTEVENTS_CV.csv` / `INPUTEVENTS_MV.csv` | Fluid/medication inputs |
| `NOTEEVENTS.csv` | Free-text clinical notes |
| `OUTPUTEVENTS.csv` | Fluid outputs |

**Target variable**: `EXPIRE_FLAG` from `PATIENTS.csv` (1 = died in hospital, 0 = survived).

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd urops-code
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download a sentence transformer model

Clinical notes are encoded using a [Sentence Transformers](https://www.sbert.net/) model. Place the model directory at `../Sent_Model` relative to `Model-Code/` (i.e., one level above the repo root), or update the path in [`patient_Dataset.py`](Model-Code/patient_Dataset.py:11).

---

## Data Preprocessing

All preprocessing scripts are in [`Data-Preprocessing/`](Data-Preprocessing/).

### Step 1 — Build the patient list

```bash
cd Data-Preprocessing
python get_SubDOB.py
```

Reads `PATIENTS.csv`, builds a subject-ID → date-of-birth mapping, and saves a sorted patient list to `../data/dataByPatient/patientList.pkl`.

### Step 2 — Save demographic category encodings

```bash
python admission_cats.py
```

Saves the unique values for each categorical demographic field (ethnicity, language, marital status, insurance, religion) as `.npy` files.

### Step 3 — Split all MIMIC-III tables by patient

```bash
python split_Data.py
```

Processes all MIMIC-III event tables in chunks, normalises timestamps relative to each patient's date of birth (in fractional years), and writes one CSV per patient per modality into `../data/dataByPatient/{Adm,Labs,Microbio,CPT,Inputs,Notes,Outputs}/`.

> **HPC users**: a PBS job script is provided at [`Data-Preprocessing/data-process.txt`](Data-Preprocessing/data-process.txt) for running preprocessing on a Volta GPU cluster via Singularity.

---

## Training

Training is managed by [`Model-Code/trainer.py`](Model-Code/trainer.py) using PyTorch Lightning and Ray Tune.

```bash
cd Model-Code
python trainer.py
```

### Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `num_layers` | 4 | Number of Transformer encoder layers per modality |
| `embedDim` | 512 | Embedding dimension |
| `finalDim` | 120 | Output dimension of the condensor layer |
| `lr` | 1e-5 | Learning rate (AdamW) |
| `wd` | 1e-5 | Weight decay |
| `beta` | 20 | Adversarial loss weight (0 = no adversarial regularisation) |
| `batch_size` | 16 | Training batch size |
| `num_epochs` | 10 | Maximum training epochs |

The ASHA scheduler is used for early stopping of underperforming trials. Results and checkpoints are saved to `~/ray_results/checkpoint/tune_MIMIC`.

### Data split

- **Train**: first 40,000 patients (shuffled with seed 42)
- **Validation**: next 4,000 patients

---

## Results

The `Results/` directory contains CSV files with per-patient embeddings, model predictions, and ground-truth labels for three values of `beta`. All models were evaluated on a held-out **test set of 2,520 patients** (patients 44,000 onward after shuffling with seed 42), of whom **1,700 survived** (class 0) and **820 died** (class 1).

Each row contains:
- `embeddings`: the 120-dimensional patient representation from the condensor
- `prediction`: the classifier's predicted mortality probability (sigmoid output)
- `actual`: the ground-truth `EXPIRE_FLAG` (0 = survived, 1 = died)

| File | Beta | Description |
|---|---|---|
| [`Results/beta0.csv`](Results/beta0.csv) | 0 | Classifier only — no adversarial regularisation (baseline) |
| [`Results/beta2.csv`](Results/beta2.csv) | 2 | Light adversarial regularisation |
| [`Results/beta20.csv`](Results/beta20.csv) | 20 | Strong adversarial regularisation |

---

### Classification Performance

| Metric | beta=0 | beta=2 | beta=20 |
|---|---|---|---|
| **AUC-ROC** | **0.820** | 0.819 | 0.777 |
| Accuracy | 0.689 | 0.599 | 0.720 |
| Precision (survived, 0) | 0.892 | 0.923 | 0.794 |
| Recall (survived, 0) | 0.613 | 0.442 | 0.790 |
| F1 (survived, 0) | 0.727 | 0.598 | 0.792 |
| Precision (died, 1) | 0.513 | 0.444 | 0.569 |
| Recall (died, 1) | 0.846 | 0.923 | 0.576 |
| F1 (died, 1) | 0.639 | 0.600 | 0.573 |
| **Macro avg F1** | **0.683** | 0.599 | 0.682 |

Training over 8 epochs (beta=2 run) showed consistent improvement: validation accuracy rose from **72.6% → 76.75%** and validation loss fell from **0.559 → 0.478**.

**Key finding**: Adversarial regularisation does **not** improve classification performance. Both beta=2 and beta=20 match or underperform the baseline (beta=0) on AUC-ROC and macro F1. Beta=20 in particular drops AUC from 0.820 to 0.777, suggesting that a very strong discriminator loss interferes with the classifier's ability to learn discriminative features.

---

### Representation Quality (Embedding Geometry)

To assess whether beta improves the *quality* of the learned representations — independent of classification accuracy — the geometry of the 120-dimensional embeddings was analysed.

| Metric | beta=0 | beta=2 | beta=20 |
|---|---|---|---|
| Mean L2 norm | 107.99 ± 57.78 | 97.60 ± 57.32 | **90.84 ± 18.73** |
| Mean per-dim std | 7.55 | 7.13 | **4.83** |
| **Effective rank** (isotropy) | 3.48 / 120 | 3.52 / 120 | **4.49 / 120** |
| Variance in top-1 PC | 62.0% | 64.1% | **51.6%** |
| Variance in top-5 PCs | 96.5% | 95.9% | **93.0%** |
| Intra-class dist (survived) | 72.68 | 70.23 | 70.89 |
| Intra-class dist (died) | 103.11 | 92.85 | **57.05** |
| Inter-class dist | 97.08 | 90.93 | 69.51 |
| Inter/Intra ratio (survived) | 1.336 | 1.295 | 0.980 |
| Inter/Intra ratio (died) | 0.942 | 0.979 | **1.218** |

**Key findings on representation quality:**

- **Beta=20 produces more isotropic embeddings**: The effective rank rises from 3.48 (beta=0) to 4.49 (beta=20), and the fraction of variance explained by the top principal component drops from 62% to 51.6%. This means the representations are less collapsed onto a single dominant direction — a sign of better-spread, more informative embeddings.

- **Beta=20 reduces norm variance**: The standard deviation of L2 norms drops dramatically from ±57.78 (beta=0) to ±18.73 (beta=20), indicating that the adversarial pressure prevents the model from producing wildly different-magnitude embeddings for different patients. More uniform norms suggest a more stable representation space.

- **Beta=20 tightens the "died" cluster**: The intra-class distance for the "died" class shrinks from 103.11 (beta=0) to 57.05 (beta=20), while the inter/intra ratio for that class improves from 0.942 to 1.218. This means patients who died are represented more compactly and are better separated from survivors in embedding space — even though the raw classification accuracy is lower.

- **The trade-off**: Beta=20 produces geometrically better-structured representations (more isotropic, more compact class clusters, more uniform norms) but at the cost of AUC-ROC (0.777 vs 0.820). This is consistent with the adversarial loss regularising the embedding space at the expense of raw discriminative power. Beta=2 offers no meaningful improvement on either axis.

- **Practical implication**: If the goal is downstream use of the embeddings (e.g., clustering, transfer learning, or fairness analysis), beta=20 may be preferable. If the goal is pure mortality prediction accuracy, beta=0 is the best choice.

---

### Interpretation

- **Strong discriminative power (AUC = 0.82)**: The multimodal Transformer achieves an AUC-ROC of 0.82 on the held-out test set with no adversarial regularisation, indicating that the fused representation captures clinically meaningful signals across all seven data modalities.

- **Class imbalance trade-off**: The dataset is imbalanced (~67% survived, ~33% died). At the default 0.5 threshold, beta=0 achieves high recall for the "died" class (0.85) at the cost of precision (0.51), erring on the side of flagging more patients as high-risk — a clinically sensible trade-off. Beta=20 produces a more balanced precision/recall profile.

- **Multimodal fusion**: By encoding clinical notes via a frozen Sentence Transformer alongside structured time-series data (labs, vitals, inputs/outputs, microbiology, procedures) and demographics, the model integrates complementary information sources that individually would be insufficient for robust mortality prediction.

Post-hoc analysis (ROC curves, confusion matrices, t-SNE plots) is in [`Data-Analysis/Post- ML Analysis.ipynb`](Data-Analysis/Post-%20ML%20Analysis.ipynb).

---

## Dependencies

```
nltk
tqdm
torch
transformers
pandas
numpy
pytorch-lightning
sentence-transformers
ray[tune]
```

Install with:

```bash
pip install -r requirements.txt
```
