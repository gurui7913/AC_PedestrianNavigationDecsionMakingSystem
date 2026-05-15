# Eye-Tracking Based Urban Navigation Decision System
**Track the Eyes, Track the Mind**

## Project Overview
This project investigates the relationship between pedestrians' 
visual attention and their wayfinding decisions in urban environments. 
Using eye-tracking experiments and multimodal machine learning, 
we built a system that predicts navigation behaviour from visual 
attention patterns.

**Research Question:** What is the relationship between pedestrians' 
visual attention distribution and the environmental cues they rely 
on for wayfinding decisions when searching for a train station?

---

## Pipeline Overview

- Eye-tracking Experiment (GazeRecorder)
- Heatmap Generation (5 participants × 13 SVIs)
- Visual Feature Extraction (CLIP ViT-B/32)
- Text Feature Extraction (CLIP, voice-to-text descriptions)
- Cosine Similarity Analysis
- Path Decision Classification (Random Forest + SMOTE)
- Heatmap Prediction (Vision Transformer)
 
---

## Methods

### Data Collection
- Selected 13 static Street View Images (SVIs) near King's Cross 
  Station, London
- Recruited 5 participants (university students, aged 22-23)
- Used GazeRecorder for webcam-based eye-tracking
- Recorded participants' navigation decisions via voice-to-text

### Feature Extraction
- **Visual features:** CLIP (ViT-B/32) encodes heatmap images 
  into 512-dimensional embeddings
- **Text features:** CLIP text encoder processes verbal path 
  descriptions into the same embedding space
- **Similarity:** Cosine similarity measures alignment between 
  visual attention and verbal reasoning

### Classification Model
- Combined visual + text features as input
- SMOTE for class imbalance (3 classes: turn left, go straight, 
  turn right)
- Random Forest classifier (200 trees, max depth 10)
- **Result: 80% accuracy** (vs 50% random baseline)

### Heatmap Prediction
- Vision Transformer (ViT) fine-tuned to predict attention 
  heatmaps from original street view images
- HSV colour space used to interpret hotspot intensity regions

---

## Key Findings
- Identified 5 primary urban elements capturing pedestrian 
  attention: **Buildings, Roads, Trees, Cars, Signs**
- Buildings received highest fixation duration on public roads
- Roads received highest fixation on private/community roads
- Visual attention patterns are significantly aligned with 
  verbal navigation reasoning (cosine similarity ~0.25)

---

## Limitations
- Small sample size (5 participants, all university students)
- Static SVIs limit dynamic navigation context
- GazeRecorder accuracy constrained by screen resolution (800×600)

## Future Work
- Replace GazeRecorder with customized OpenCV eye-tracking
- Expand to interactive 3D navigation environment
- Diversify participant demographics

---

## Tech Stack
- Python, PyTorch
- OpenAI CLIP (ViT-B/32)
- scikit-learn (Random Forest)
- imbalanced-learn (SMOTE)
- OpenCV, NumPy, Pandas, Matplotlib

## Dependencies
```bash
pip install torch torchvision
pip install git+https://github.com/openai/CLIP.git
pip install scikit-learn imbalanced-learn
pip install opencv-python numpy pandas matplotlib
```

---

## Project Structure
├── extract_visual_attention.py    # HSV-based heatmap processing
├── highlight_visual_differences.py # Original vs heatmap comparison
├── image_text_feature_analysis.py  # CLIP image feature extraction
├── text_feature_extraction.py      # CLIP text feature extraction
├── feature_similarity_analysis.py  # Cosine similarity computation
├── train_path_choice_model_en.py   # Random Forest classifier
├── visualize_similarity.py         # Results visualization
└── README.md

---

## Team
- **Rui Gu** — Project Lead, Research Design, Technical Direction, Experiment Coordination
 - Hexin Han — ML pipeline Implementation  

*UCL Architectural Computation Studio 1, December 2024*
