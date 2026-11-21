# Aquarium Object Detection with YOLOv8n

**CSCI1409 Deep Learning — Bonus Project**

## Project overview
Fine-tune a pretrained YOLOv8n object detector to detect seven aquatic-animal classes. The Google Colab notebook downloads the dataset from Kaggle, audits annotations, trains four configurations, selects the final checkpoint using validation results, evaluates the held-out test split, and creates visualizations.

## Dataset
- **Kaggle:** [Aquarium Data COTS](https://www.kaggle.com/datasets/slavkoprytula/aquarium-data-cots)
- **Background:** 638 images collected by Roboflow from Henry Doorly Zoo, Omaha (October 16, 2020) and the National Aquarium, Baltimore (November 14, 2020).
- **Annotations:** Roboflow team, with some help from SageMaker Ground Truth.
- **License:** Creative Commons Attribution (CC BY), according to the supplied dataset description. Acknowledge the source when reusing the images and annotations.
- **Format:** YOLO text format, one object per line: `class_id x_center y_center width height`; coordinates are normalized to [0, 1].

### Classes
| ID | Class |
|---:|---|
| 0 | fish |
| 1 | jellyfish |
| 2 | penguin |
| 3 | puffin |
| 4 | shark |
| 5 | starfish |
| 6 | stingray |

### Dataset splits
| Split | Images | Valid objects |
|---|---:|---:|
| Train | 448 | 3,324 |
| Validation | 127 | 909 |
| Test | 63 | 582 |
| **Total** | **638** | **4,815** |

The notebook's pairing audit found 448/448 train, 127/127 validation, and 63/63 test image-label pairs. The annotation audit flagged **two invalid bounding-box dimension entries** in test labels. These are reported, not silently changed. The recorded Ultralytics test evaluation reported 584 instances, differing from the 582 valid objects counted by the audit; investigate this if rerunning or modifying labels.

## Model and environment
- Framework: Ultralytics YOLO `8.4.216`
- Model: YOLOv8n, pretrained weights `yolov8n.pt`
- Recorded runtime: Python 3.13.16, PyTorch 2.14.1+cu130, Tesla T4 GPU
- Final image size: 512 × 512
- Batch size: 16
- Random seed: 42
- Final training configuration: 30 epochs, patience 10
- Notebook working directory: `/content/aquarium_project/`

## Experiments
| Run | Image size | Epochs | Validation precision | Validation recall | Validation mAP@0.50 | Time |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 320 | 5 | 47.0% | 49.1% | 46.9% | 0.86 min |
| Experiment 2 | 416 | 5 | 50.4% | 51.8% | 50.7% | 0.73 min |
| Experiment 3 | 416 | 30 | 74.6% | 64.6% | 69.5% | 3.31 min |
| Experiment 4 | 512 | 30 | 74.4% | 71.1% | 74.1% | 3.61 min |

The 512-pixel, 30-epoch checkpoint was selected using validation performance. The 416/30 versus 512/30 comparison is the direct image-size comparison. The overall experiment sequence also changes training duration between runs.

## Final held-out test results
Checkpoint: `runs/experiment_512/weights/best.pt`  
Evaluation: test split, image size 512, batch 16, confidence threshold 0.20.

| Metric | Result |
|---|---:|
| Precision | 0.8255 |
| Recall | 0.6663 |
| mAP@0.50 | 0.6835 |
| mAP@0.50:0.95 | 0.4334 |
| mAP@0.75 | 0.4823 |

### Per-class test AP@0.50
| Class | AP@0.50 |
|---|---:|
| Fish | 58.5% |
| Jellyfish | 81.0% |
| Penguin | 61.2% |
| Puffin | 42.5% |
| Shark | 75.6% |
| Starfish | 87.2% |
| Stingray | 72.4% |

## Run in Google Colab
1. Open `aquarium_object_detection_yolov8.ipynb` in Google Colab.
2. Select **Runtime → Change runtime type → GPU**.
3. Run all cells from top to bottom.
4. Complete Kaggle authentication if prompted. Never commit `kaggle.json`, API tokens, or credentials.
5. The notebook installs packages, downloads the dataset, prepares the YAML file, trains, evaluates, and generates visualizations.

After publishing this repository, add an Open in Colab badge using the repository's actual notebook URL:
```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_GITHUB_NOTEBOOK_URL)
```

## Repository contents
- `aquarium_object_detection_yolov8.ipynb` — complete Colab workflow.
- `requirements.txt` — Python dependencies.
- `configs/` — dataset configuration notes.
- `results/` — training logs, curves, and experiment evidence.
- `visualizations/` — confusion matrices and prediction images.
- `weights/` — final checkpoint or external download instructions.

**Before submission:** copy your actual run artifacts into the matching folders. Upload `best.pt` if it is within GitHub's file-size limit; otherwise host it on Drive or another accessible service and add a working instructor-accessible download link here. Do not leave the final weights link blank.

## Limitations and failure analysis
The model has lower recall for fish and puffin than for several other classes. Visual inspection showed missed small/distant objects, overlapping detections in crowded scenes, and predictions without matching annotations. Possible causes include object scale, occlusion, background complexity, class imbalance, and annotation quality. A practical next step is to review the flagged labels and add/improve examples for difficult classes.

## Attribution
Dataset: Roboflow Aquarium dataset, accessed through [Kaggle](https://www.kaggle.com/datasets/slavkoprytula/aquarium-data-cots). The supplied dataset description attributes image collection to Henry Doorly Zoo and National Aquarium and annotation to Roboflow with some help from SageMaker Ground Truth. Reuse is subject to the stated CC BY attribution requirement.

Framework: [Ultralytics](https://docs.ultralytics.com/).

## Assistance
This project uses the Ultralytics detection framework and publicly available pretrained YOLOv8n weights. The student should understand and be prepared to explain data preparation, annotations, training settings, experiments, metrics, and failure cases during the individual defense.
