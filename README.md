# Track the Eyes, Track the Mind
**Eye-Tracking Based Urban Navigation Decision System**

> UCL Bartlett School of Architecture · Architectural Computation – Digital Studio 1: Simulated Realities  
> Team: **Rui Gu** (Lead), Hexin Han, Cem Bektas · December 2024

> **My contribution:** Research design · Eye-tracking experiment coordination · CLIP pipeline · Cosine similarity analysis · Random Forest classifier · ViT heatmap predictor (architecture)

---

## Table of Contents

- [Research Question](#research-question)
- [System Architecture](#system-architecture)
  - [Data Collection](#data-collection)
  - [Feature Extraction — CLIP](#feature-extraction--clip)
  - [Cross-Modal Alignment Analysis](#cross-modal-alignment-analysis)
  - [Path Decision Classification](#path-decision-classification)
  - [Heatmap Prediction — ViT (Proposed)](#heatmap-prediction--vit-proposed)
- [Workflow](#workflow)
  - [Stage 1 — Eye-Tracking Experiment](#stage-1--eye-tracking-experiment)
  - [Stage 2 — Feature Extraction](#stage-2--feature-extraction)
  - [Stage 3 — Cosine Similarity Analysis](#stage-3--cosine-similarity-analysis)
  - [Stage 4 — Path Decision Classification (My Core Contribution)](#stage-4--path-decision-classification-my-core-contribution)
  - [Stage 5 — Heatmap Prediction](#stage-5--heatmap-prediction)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup & How to Run](#setup--how-to-run)
- [Known Limitations](#known-limitations)
- [References](#references)

---

## Research Question

Standard wayfinding research measures either where people look (eye-tracking) or what they say (verbal protocols) — rarely both, and rarely in the same computational framework. This raises a central question:

> **What is the relationship between pedestrians' visual attention distribution and the environmental cues they rely on for wayfinding decisions when searching for a train station in static Street View Images?**

We operationalize this through two sub-questions:

1. **Alignment:** Can CLIP's shared embedding space quantify semantic agreement between gaze heatmaps and verbal reasoning? (→ cosine similarity analysis)
2. **Prediction:** Do fused gaze + language features reliably predict discrete path decisions (Left / Straight / Right)? (→ Random Forest classifier)

**Motivating scenario:** A pedestrian in an unfamiliar city, mobile phone dead, must navigate to a train station using only visual environmental cues. Which urban elements do they look at? Does what they look at match what they say?

---

## System Architecture

```
Raw Data
 ├── Eye-tracking sessions (GazeRecorder, webcam-based)
 │     5 participants × 13 SVIs = 65 gaze recordings
 │     Output: heatmap PNG per (participant, SVI), 800×600
 └── Voice-to-text logs
       65 verbal navigation descriptions
       Output: path label (L/S/R) + reasoning string per (participant, SVI)

Stage 1 — Feature Extraction
 ├── Heatmap → CLIP image encoder
 │     RGB → HSV conversion (H channel = fixation intensity gradient)
 │     Resize to 224×224
 │     CLIP ViT-B/32, frozen → 512-d visual embedding
 │     Output: visual_embeddings.npy  [65, 512]
 │
 └── Verbal description → CLIP text encoder
       BPE tokenisation → CLIP text encoder, frozen → 512-d text embedding
       (same embedding space as image encoder via contrastive pre-training)
       Output: text_embeddings.npy    [65, 512]

Stage 2 — Alignment Analysis
 └── Pairwise cosine similarity: visual_emb[i] · text_emb[i]
       Output: similarity_scores.csv  [65 rows: participant_id, svi_id, cosine_sim]

Stage 3 — Path Decision Classification
 └── [visual_emb ∥ text_emb] → 1024-d concatenation
       → SMOTE oversampling (65 → 69 samples, 3 balanced classes)
       → Random Forest (n_estimators=200, max_depth=10, 3-fold CV)
       Output: model.pkl, classification_report.txt

Stage 4 — Heatmap Prediction (proposed; training not completed)
 └── Original SVI (224×224×3) → ViT backbone → custom decoder → heatmap (224×224)
```

---

## System Architecture Details

### Data Collection

**Study site:** 13 static Street View Images along the pedestrian approach to King's Cross Station, London (8 main roads, 5 community/residential roads) + 4 warm-up images (excluded from analysis).

**Participants:** N=5, aged 22–23, university students. Each participant viewed all 13 SVIs sequentially and made a forced-choice path decision (Left / Straight / Right) while narrating their reasoning aloud.

**Eye-tracking instrument:** [GazeRecorder](https://gazerecorder.com/) — browser-based, webcam-only, no dedicated hardware required.

| Parameter | Value |
|:---|:---|
| Calibration | 9-point facial landmark mapping |
| Output format | PNG heatmap per (participant, SVI), 800×600 px |
| Gaze mapping | Facial landmark regression → screen coordinates |
| Raw coordinates | Not exported; heatmap rendering is GazeRecorder's output |

**Why GazeRecorder over Tobii / SMI hardware:** Zero hardware cost; sufficient for an exploratory pilot study. Acknowledged limitation: no calibration validation, no raw gaze coordinate access, 800×600 resolution cap. A hardware-based system (e.g. Tobii Pro Nano) would be used in a scaled follow-up.

**Label derivation:** Voice-to-text transcripts were manually reviewed. Each transcript was assigned one of three labels (L / S / R) based on the participant's stated path choice, then normalised (lowercased, punctuation removed) for CLIP encoding.

| Class | Count (raw) | After SMOTE |
|:---:|:---:|:---:|
| Left | 20 | 23 |
| Straight | 28 | 23 |
| Right | 17 | 23 |
| **Total** | **65** | **69** |

---

### Feature Extraction — CLIP

We use OpenAI CLIP (ViT-B/32) as a **frozen feature extractor** — it is not fine-tuned at any stage. CLIP's contrastive pre-training on 400M image-text pairs places semantically related images and texts near each other in a shared 512-d embedding space, enabling direct cosine comparison between a gaze heatmap and its corresponding verbal description without task-specific supervision.

**Why CLIP over a task-specific encoder?**

- No paired (heatmap, description) training data exists at this scale — fine-tuning is not feasible with N=65
- CLIP's semantic embedding space captures object-level and scene-level semantics that are directly relevant to navigation (buildings, roads, signs)
- A CNN-based image encoder trained on image classification would not share a representation space with text, making cross-modal comparison impossible

**Visual branch — HSV conversion rationale:**

Raw heatmaps encode fixation intensity as colour (red = high, blue = low in the default GazeRecorder palette). Passing RGB heatmaps directly to CLIP would conflate colour semantics with fixation intensity. Converting to HSV isolates fixation intensity in the **H (hue) channel**, separating it from brightness (V) and saturation (S). CLIP then encodes the H channel pattern as a structured spatial distribution.

```python
import cv2
import clip
import torch
import numpy as np

model, preprocess = clip.load("ViT-B/32", device="cpu")
model.eval()

def extract_visual_embedding(heatmap_path: str) -> np.ndarray:
    """
    Load a GazeRecorder heatmap PNG, convert RGB→HSV,
    encode with CLIP ViT-B/32 (frozen).
    Returns: numpy array of shape [512]
    """
    img_bgr = cv2.imread(heatmap_path)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Convert single-channel H back to 3-channel for CLIP's RGB input
    img_h = img_hsv[:, :, 0]
    img_rgb = cv2.cvtColor(
        cv2.merge([img_h, img_h, img_h]), cv2.COLOR_GRAY2RGB
    )
    pil_img = Image.fromarray(img_rgb)
    tensor = preprocess(pil_img).unsqueeze(0)
    with torch.no_grad():
        embedding = model.encode_image(tensor)
    return embedding.squeeze().numpy()   # shape: [512]
```

**Text branch:**

```python
def extract_text_embedding(description: str) -> np.ndarray:
    """
    Tokenise a verbal navigation description with CLIP's BPE tokeniser,
    encode with CLIP text encoder (frozen).
    Returns: numpy array of shape [512]
    """
    tokens = clip.tokenize([description])
    with torch.no_grad():
        embedding = model.encode_text(tokens)
    return embedding.squeeze().numpy()   # shape: [512]
```

---

### Cross-Modal Alignment Analysis

We measure alignment between visual attention (heatmap embedding) and verbal reasoning (text embedding) using cosine similarity:

$$\text{sim}(v, t) = \frac{v \cdot t}{\|v\| \cdot \|t\|}$$

where $v \in \mathbb{R}^{512}$ is the visual embedding and $t \in \mathbb{R}^{512}$ is the text embedding for the same (participant, SVI) pair.

This is computed for all 65 pairs. Key properties of the measure:

- **Range:** $[-1, +1]$; random unit vectors in 512-d space have expected cosine $\approx 0$
- **Interpretation:** Values > 0 indicate that the attended visual regions and the verbal description share semantic content in CLIP's representation space
- **Limitation:** CLIP encodes general visual semantics, not task-specific navigation relevance. A low score does not mean the participant was looking at the wrong thing — it means gaze and language diverge in *CLIP's* semantic space

---

### Path Decision Classification

**Why Random Forest over neural classifiers?**

With N=65 (69 after SMOTE), any network with learnable fusion weights will overfit. Random Forest is the appropriate choice:

- Non-parametric; no gradient-based overfitting
- Native handling of high-dimensional inputs (1024-d) relative to sample count
- Feature importance scores provide interpretability (which embedding dimensions drive decisions)
- Ensemble averaging reduces variance from the small dataset

**Why simple concatenation over cross-attention fusion?**

A cross-attention or gating layer would require ≥500 paired samples to generalise. Concatenation [visual ∥ text] treats both modalities equally — the Random Forest then implicitly weights dimensions via feature selection at each tree split. This is the most defensible approach given N=65.

**Feature vector construction:**

$$x_i = [v_i \| t_i] \in \mathbb{R}^{1024}$$

where $v_i \in \mathbb{R}^{512}$ is the visual embedding and $t_i \in \mathbb{R}^{512}$ is the text embedding for sample $i$.

**Class imbalance — SMOTE:**

With 20/28/17 class distribution, a classifier trained on raw data would be biased toward "Straight." SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic samples by interpolating between k nearest neighbours in feature space:

$$x_{\text{new}} = x_i + \lambda \cdot (x_{k\text{-nn}} - x_i), \quad \lambda \sim \mathcal{U}(0, 1)$$

Parameters: `k_neighbors=3`, `random_state=42`. Applied only to the training fold during cross-validation to prevent data leakage.

```python
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
import numpy as np

visual = np.load("outputs/visual_embeddings.npy")   # [65, 512]
text   = np.load("outputs/text_embeddings.npy")     # [65, 512]
X = np.concatenate([visual, text], axis=1)          # [65, 1024]
# labels: np.load("data/labels.npy")  — 0=Left, 1=Straight, 2=Right

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

smote = SMOTE(k_neighbors=3, random_state=42)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

cv_scores = []
for train_idx, val_idx in cv.split(X, y):
    X_res, y_res = smote.fit_resample(X[train_idx], y[train_idx])
    rf.fit(X_res, y_res)
    cv_scores.append(rf.score(X[val_idx], y[val_idx]))

print(f"CV accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
```

---

### Heatmap Prediction — ViT (Proposed)

**Task:** Given an original SVI (no eye-tracking data), predict the average gaze heatmap across participants — i.e., predict *where pedestrians would look* from scene appearance alone.

**Why ViT over CNN:**

A CNN's local receptive field means each feature captures only a small spatial patch. Gaze fixations are globally distributed — a pedestrian may look at a distant building, a near road sign, then back to the building. ViT's **self-attention mechanism** captures these cross-region dependencies by computing attention weights between all pairs of patch tokens:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\T}{\sqrt{d_k}}\right)V$$

This global receptive field better mirrors the spatial structure of fixation patterns than local convolutions.

**Architecture:**

```
Input SVI: [B, 3, 224, 224]
    │
    ▼
ViT Backbone (ImageNet pretrained, patch size 16, 196 patches)
    │   → CLS token + 196 patch tokens
    │   → [B, 197, 768]  (ViT-Base)
    ▼
Global Average Pool over patch tokens → [B, 768]
    │
    ▼
Decoder:
    Linear(768 → 512) → ReLU
    Linear(512 → 224×224) → Sigmoid
    │
    ▼
Predicted heatmap: [B, 1, 224, 224]
    reshape + normalise to [0, 1]
```

| Parameter | Value |
|:---|:---|
| Backbone | ViT-Base/16 (ImageNet pretrained) |
| Decoder | Linear(768→512) → ReLU → Linear(512→50176) → Sigmoid |
| Loss | MSE between predicted and averaged ground-truth heatmap |
| Optimizer | Adam, lr=0.001 |
| Batch size | 4 |
| Epochs | 20 |
| Status | Architecture implemented; training not completed (N=65 insufficient for generalisation) |

**Note on training feasibility:** With 13 unique SVIs and N=5 participants, the available training set (13 images → 13 averaged heatmaps) is far too small to train a ViT decoder from scratch. A meaningful evaluation would require ~500–1000 SVI–heatmap pairs. The architecture is documented here as a research trajectory for follow-up work.

---

## Workflow

```
Stage 1          Stage 2             Stage 3             Stage 4             Stage 5
Eye-Tracking  ─► Feature         ─► Cosine           ─► Path Decision  ─► Heatmap
Experiment       Extraction          Similarity           Classification     Prediction
                                     Analysis             (RF + SMOTE)       (ViT, WIP)

• GazeRecorder   • Heatmap→HSV       • Per-pair           • Concat           • SVI → ViT
• 5P × 13SVI     • CLIP ViT-B/32       cosine sim           embeddings         backbone
• Voice-to-text  • Text→BPE→CLIP     • Road type          • SMOTE            • Custom
• Labels L/S/R   • 512-d vectors       breakdown          • 3-fold CV          decoder
```

### Stage 1 — Eye-Tracking Experiment

Recruited 5 participants; calibrated GazeRecorder per-session (9-point). Participants viewed SVIs full-screen (1920×1080 display; GazeRecorder captures at 800×600). Each SVI was displayed for an unconstrained duration; participants indicated their path decision (L/S/R) verbally. Sessions were recorded via screen capture + audio.

Post-session processing:
- GazeRecorder exports one heatmap PNG per (participant, SVI)
- Audio transcribed via browser speech-to-text API; manually reviewed for accuracy
- Labels assigned from transcript; reasoning strings extracted verbatim

### Stage 2 — Feature Extraction

```bash
# Visual embeddings from heatmaps
python extract_visual_attention.py \
    --heatmap_dir data/heatmaps \
    --output outputs/visual_embeddings.npy

# Text embeddings from verbal descriptions
python text_feature_extraction.py \
    --transcript_dir data/transcripts \
    --output outputs/text_embeddings.npy
```

Both scripts use the same frozen CLIP ViT-B/32 model instance. Embeddings are L2-normalised before saving (unit vectors; cosine similarity reduces to dot product).

### Stage 3 — Cosine Similarity Analysis

```bash
python feature_similarity_analysis.py \
    --visual outputs/visual_embeddings.npy \
    --text   outputs/text_embeddings.npy \
    --labels data/labels.csv \
    --output outputs/similarity_scores.csv
```

Outputs per-pair similarity scores and breakdowns by:
- Road type (public main road vs community road)
- Path decision class (L / S / R)
- SVI index (scene-level variation)

### Stage 4 — Path Decision Classification (My Core Contribution)

```bash
python train_path_choice_model_en.py \
    --visual   outputs/visual_embeddings.npy \
    --text     outputs/text_embeddings.npy \
    --labels   data/labels.csv \
    --output   outputs/
```

Prints CV accuracy and test accuracy; writes `outputs/model.pkl` and `outputs/classification_report.txt`.

### Stage 5 — Heatmap Prediction

```bash
python heatmap_predictor.py \
    --svi_dir    data/svi \
    --heatmap_dir data/heatmaps \
    --output     outputs/predicted_heatmaps/
```

**Status:** Architecture trains without error; MSE does not converge meaningfully with N=13 training images. Listed here for reproducibility of the architecture; a larger SVI–heatmap dataset is required for valid evaluation.

---

## Results

### Quantitative

| Metric | Value | Baseline |
|:---|:---:|:---:|
| Test set accuracy (RF) | **80%** | 33.3% (random 3-class) |
| 3-fold CV accuracy | **86.96% ± ~3%** | — |
| Mean cosine similarity (gaze ↔ language) | **~0.25** | ~0.00 (random vectors) |
| Cosine range across 13 SVIs | 0.22–0.27 | — |

### Attention Element Breakdown

| Road Type | Rank 1 Fixation Target | Rank 2 |
|:---|:---:|:---:|
| Public (main) road | **Buildings** | Roads |
| Community (side) road | **Roads** | Buildings |

Buildings act as implicit navigation anchors on main roads — not formal wayfinding signage. On community roads where facades are less distinctive, road geometry becomes the primary cue. This **context-dependent attention hierarchy** is the study's key empirical finding.

### Interpretation of Cosine ~0.25

A mean cosine of ~0.25 (range 0.22–0.27) across all 65 pairs indicates:

- **Non-zero alignment:** What participants look at and what they say share semantic content in CLIP's embedding space — ruling out the null hypothesis of no relationship
- **Substantial gap from perfect alignment (1.0):** Verbal reasoning incorporates factors inaccessible to the eye-tracking heatmap: spatial memory, prior knowledge of the city, sequential context from earlier frames, and inferred structure beyond the immediate visual field

This gap directly motivates the PhD research question: do LLM-based agents, which reason explicitly through language, show tighter alignment between their "attended" visual regions and their stated navigation reasoning?

### Emergent Findings

| Finding | Implication |
|:---|:---|
| 80% classification accuracy from fused gaze + language features | CLIP multimodal features carry sufficient signal for path prediction at this scale |
| Buildings dominate fixation on main roads | Implicit landmark hierarchy — buildings over formal signage |
| Context shift to roads on community roads | Attention strategy adapts to environmental affordances |
| Cosine ~0.25 (positive but weak) | Verbal reasoning draws on non-visual cognitive factors |

---

## Project Structure

```
.
├── data/
│   ├── svi/                              # 13 Street View Images (PNG, 640×640)
│   │   └── svi_{01-13}.png
│   ├── heatmaps/                         # GazeRecorder exports (PNG, 800×600)
│   │   └── participant_{1-5}/
│   │       └── svi_{01-13}_heatmap.png
│   ├── transcripts/                      # Voice-to-text (TXT, one per participant×SVI)
│   │   └── participant_{1-5}/
│   │       └── svi_{01-13}.txt
│   └── labels.csv                        # columns: participant_id, svi_id, label, description
│
├── outputs/                              # Generated by running the pipeline
│   ├── visual_embeddings.npy             # [65, 512] float32
│   ├── text_embeddings.npy               # [65, 512] float32
│   ├── similarity_scores.csv             # participant_id, svi_id, cosine_sim
│   ├── model.pkl                         # trained Random Forest
│   ├── classification_report.txt
│   └── predicted_heatmaps/              # Stage 5 output (WIP)
│
├── extract_visual_attention.py           # Stage 2a: heatmap PNG → HSV → CLIP embedding
├── image_text_feature_analysis.py        # Stage 2a (alt): original SVI → CLIP image embedding
├── highlight_visual_differences.py       # Visualisation: original SVI vs heatmap overlay
├── text_feature_extraction.py            # Stage 2b: verbal description → CLIP text embedding
├── feature_similarity_analysis.py        # Stage 3: pairwise cosine similarity
├── train_path_choice_model_en.py         # Stage 4: SMOTE + Random Forest classifier
├── visualize_similarity.py               # Plots: similarity distribution, per-SVI breakdown
├── heatmap_predictor.py                  # Stage 5 (WIP): ViT heatmap prediction
│
├── requirements.txt
└── README.md
```

---

## Setup & How to Run

### Prerequisites

```
python == 3.10
torch == 2.1.0
torchvision == 0.16.0
ftfy == 6.1.1
regex == 2023.10.3
tqdm == 4.66.1
# CLIP has no PyPI release — install from source (see below)
scikit-learn == 1.3.2
imbalanced-learn == 0.11.0
opencv-python == 4.8.1.78
numpy == 1.26.2
pandas == 2.1.3
matplotlib == 3.8.2
scipy == 1.11.4
Pillow == 10.1.0
timm == 0.9.12   # for ViT backbone (Stage 5 only)
```

### Installation

```bash
# 1. Clone
git clone https://github.com/<your-repo>/track-the-eyes.git
cd track-the-eyes

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git

# 4. Verify
python -c "import clip; m, _ = clip.load('ViT-B/32'); print('CLIP OK —', sum(p.numel() for p in m.parameters()), 'params')"
```

> **GPU:** Not required. CLIP inference on CPU: ~3s/image. RF training: <5s total.

### Run the Full Pipeline

```bash
# Stage 2 — extract embeddings
python extract_visual_attention.py --heatmap_dir data/heatmaps --output outputs/visual_embeddings.npy
python text_feature_extraction.py  --transcript_dir data/transcripts --output outputs/text_embeddings.npy

# Stage 3 — alignment analysis
python feature_similarity_analysis.py \
    --visual outputs/visual_embeddings.npy \
    --text   outputs/text_embeddings.npy \
    --labels data/labels.csv \
    --output outputs/similarity_scores.csv

# Stage 4 — classifier
python train_path_choice_model_en.py \
    --visual  outputs/visual_embeddings.npy \
    --text    outputs/text_embeddings.npy \
    --labels  data/labels.csv \
    --output  outputs/

# Visualise
python visualize_similarity.py --scores outputs/similarity_scores.csv --labels data/labels.csv
```

---

## Known Limitations

| Limitation | Technical detail | Impact |
|:---|:---|:---|
| N=5 participants | All students, aged 22–23, single demographic | Results cannot generalise; fixation patterns likely biased toward architecture-trained visual attention |
| GazeRecorder accuracy | Webcam-based, no hardware calibration, 800×600 cap | Heatmap boundary noise propagates into CLIP embeddings; fixation centroids may be offset by 1–3° visual angle |
| Simple feature concatenation | Equal weighting; no learnable fusion | Relative contribution of gaze vs language signal is uncontrolled; a gating mechanism would require ≥500 samples |
| Static SVIs | No temporal sequence, no depth cues, no peripheral vision | Real navigation is sequential and embodied; static frames capture a single viewpoint moment |
| SMOTE on N=65 | 3-NN interpolation; minimal diversity gain | Synthetic samples lie on straight lines between real samples; risk of in-distribution clustering |
| ViT heatmap predictor | N=13 training images | MSE does not converge; architecture is a research design, not a trained model |

---

## References

1. Radford, A. et al. "Learning Transferable Visual Models From Natural Language Supervision." *ICML 2021*. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
2. Dosovitskiy, A. et al. "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale." *ICLR 2021*. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
3. Chawla, N.V. et al. "SMOTE: Synthetic Minority Over-sampling Technique." *JAIR 16*, 2002.
4. Passini, R. "Wayfinding: Backbone of Graphic Support Systems." in *Visual Information For Everyday Use*, Taylor & Francis, 1998.
5. Zomer, L.B. et al. "Determinants of urban wayfinding styles." *Travel Behaviour and Society* 17, 2019.
6. Chen, X. et al. "Evaluating pedestrian wayfinding behaviour in day and night environments." *IEEE Access* 11, 2023.

---

## Citing This Work

```bibtex
@misc{gu2024trackeyes,
  title  = {Track the Eyes, Track the Mind:
             Eye-Tracking Based Urban Navigation Decision System},
  author = {Gu, Rui and Han, Hexin and Bektas, Cem},
  year   = {2024},
  note   = {UCL Bartlett School of Architecture,
             Architectural Computation Studio 1}
}
```
