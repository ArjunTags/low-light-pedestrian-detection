# Low-Light Pedestrian Detection

A low-light pedestrian detection pipeline using Zero-DCE image enhancement and YOLO11 (`yolo11n.pt`), trained on the visible-light portion of the LLVIP dataset.

## Project Overview

Pedestrian detection becomes difficult in nighttime and low-light scenes because image brightness, contrast, and visible detail are reduced.

This project applies the following pipeline:

```text
LLVIP visible images
        ↓
Zero-DCE low-light enhancement
        ↓
Pascal VOC XML annotations
        ↓
YOLO annotation conversion
        ↓
Train/validation/test split
        ↓
YOLO11 pedestrian detection
        ↓
Evaluation and reporting
```

## Object

The detected object is:

```text
person/pedestrian
```

## Dataset

This project uses the visible-light images from the LLVIP dataset.

The dataset is not included in this repository. Download it from the original dataset source or the source documented in the notebook.

Dataset statistics:

| Split | Images |
|---|---:|
| Original training images | 12,025 |
| Validation images | 2,405 |
| Final training images | 9,620 |
| Test images | 3,463 |
| Total visible images | 15,488 |

## Methods

### Low-Light Enhancement

- Zero-DCE
- Pretrained `Epoch99.pth` weights
- RGB visible-light images
- Original image dimensions preserved after enhancement

### Object Detection

- YOLO11 (`yolo11n.pt`)
- Image size: 640 × 640
- Epochs: 50
- Batch size: 4
- Dataset class: `person`
- TensorBoard logging enabled in notebook workflow

## Repository Contents

```text
.
├── assets/
│   ├── 220185_zerodce.jpg
│   └── 220185_yolo_box.jpg
├── notebooks/
│   └── CO1_1_LLI_Pedestrian.ipynb
├── data.yaml
├── requirements.txt
├── README.md
└── .gitignore
```

## Running the Notebook

### 1. Clone the repository

```bash
git clone https://github.com/ArjunTags/low-light-pedestrian-detection.git
cd low-light-pedestrian-detection
```

### 2. Create a virtual environment

```bash
python3 -m venv env
source env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download LLVIP

Follow the download cell in:

```text
notebooks/CO1_1_LLI_Pedestrian.ipynb
```

Place the extracted dataset at:

```text
data/llvip/LLVIP/
```

The expected structure is:

```text
data/llvip/LLVIP/
├── Annotations/
├── visible/
│   ├── train/
│   └── test/
└── infrared/
```

### 5. Open the notebook

```bash
jupyter notebook
```

Open:

```text
notebooks/CO1_1_LLI_Pedestrian.ipynb
```

## Reproducibility Notes

The notebook contains the complete process for:

- dataset acquisition;
- dataset verification;
- Zero-DCE enhancement;
- XML-to-YOLO conversion;
- dataset splitting;
- YOLO11 training using `yolo11n.pt`;
- validation and held-out test evaluation;
- confusion matrix generation (`confusion_matrix.png`, `confusion_matrix_normalized.png`);
- TensorBoard launch instructions for YOLO11 logs;
- sample prediction visualization.

Generated model outputs are written under:

```text
/home/arjun/lli-pedestrian/runs/
```

These run artifacts are not tracked in this repository by default.

## References

- LLVIP dataset: low-light visible-infrared paired pedestrian dataset
- Zero-DCE: low-light image enhancement
- Ultralytics YOLO11: object detection framework
