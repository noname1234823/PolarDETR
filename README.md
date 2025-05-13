# PolarDETR: Anatomical Entity Detection and Localization in Dental Images

PolarDETR is a deep learning framework for precise detection and localization of anatomical entities in dental images based on text descriptions. The system uses a polar coordinate encoding approach combined with anatomical constraints for improved accuracy and interpretability.

## Core Features

- **Polar Text-Position Encoding (PTPE)**: Maps textual descriptions to polar coordinates
- **Anatomical Constraint Learning**: Incorporates domain knowledge about anatomical regions
- **Position Matching Optimization**: Aligns predicted regions with text descriptions
- **Interpretability Module**: Provides anatomical consistency and position matching scores
- **FDI Tooth Notation Support**: Uses the international FDI tooth numbering system

## Project Structure

```
PolarDETR/
├── configs/             # Configuration files
├── data/                # Data preprocessing and loading utilities
│   └── FDI_MATCH.xml    # FDI tooth notation to angle mapping
├── models/              # Model architecture components
│   ├── decoders/        # DETR decoder with position enhancements
│   └── encoders/        # Text and image encoders
├── utils/               # Helper functions and evaluation metrics
│   ├── fdi_parser.py    # FDI tooth notation parser
│   ├── metrics.py       # Evaluation metrics
│   └── visualization.py # Visualization utilities
├── main.py              # Main training script
├── inference.py         # Inference script
├── demo.py              # Demo script for PTPE visualization
├── requirements.txt     # Dependencies
└── README.md            # This file
```

## Installation

```bash
git clone https://github.com/username/PolarDETR.git
cd PolarDETR
pip install -r requirements.txt
```

## Usage

### Training

```bash
python main.py --config configs/default.yaml
```

### Inference

```bash
python inference.py --model_path checkpoints/model.pth --image path/to/image.dcm --text "3mm cyst distal to tooth 37"
```

### Demo

```bash
python demo.py --text "3mm cyst distal to tooth 37" --fdi_xml data/FDI_MATCH.xml
```

## Technical Details

### FDI Tooth Notation

The system uses the FDI (Fédération Dentaire Internationale) tooth numbering system:

- First digit indicates the quadrant (1-4)
- Second digit indicates the tooth position in the quadrant (1-8)
- Example: 37 = lower left second molar

### Polar Text-Position Encoding (PTPE)

The PTPE module converts textual descriptions of anatomical locations into polar coordinates with the following process:

1. **Entity Extraction**: Extract anatomical entities (quadrant, tooth number, distance, direction)
2. **Polar Mapping**: Map entities to polar coordinates (r, θ)
3. **Position Encoding**: Generate a 3D encoding [sin(θ), cos(θ), log(r+1)]

### Evaluation Metrics

- **Anatomical Consistency Score (ACS)**: Measures alignment with anatomical regions
- **Position Matching Score (PMS)**: Measures alignment with text-derived positions
- **Standard Detection Metrics**: mAP, Precision, Recall

## Citation

If you use this code in your research, please cite:

```
@article{PolarDETR2025,
  title={PolarDETR: Enhancing Interpretability in Multi-modal Methods for Jawbone Lesion Detection in CBCT},
  author={Author, A. and Author, B.},
  ///journal={Journal of Medical Imaging},
  ///year={2025}
}
```
