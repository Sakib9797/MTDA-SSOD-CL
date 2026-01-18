# Multi-Target Domain Adaptation for Semi-Supervised Object Detection

PyTorch code for multi-domain object detection across weather conditions (normal, foggy, rainy) using:
- Dynamic curriculum learning (progressively introduce harder domains)
- Domain adaptation (adversarial domain classifier + simple feature alignment)
- Semi-supervised pseudo-labeling

This repo is organized around runnable scripts (train/eval/inference) rather than a packaged library.

## What’s in the repo

Main entry points:
- `run_pipeline.py` — runs install → dataset prep (if missing) → training → evaluation → plots
- `train.py` — training loop with curriculum scheduler and domain adaptation losses
- `evaluate.py` — evaluation across all domains
- `inference.py` — inference on images

Core modules:
- `model.py` — domain adaptation modules + curriculum scheduler + pseudo-label generator
- `prepare_dataset.py` — converts Cityscapes-style annotations into YOLO-style dataset structure
- `utils.py` — plotting/export helpers

Configuration:
- `config.yaml` — high-level defaults (model/training/curriculum)
- `requirements.txt` — Python dependencies

## Model backbone (YOLO)

Training supports two loading paths:
- Default: YOLOv11 small via Ultralytics (`yolo11s.pt`) when `model_size` contains `yolo11`/`v11`
- Fallback: YOLOv11s via `torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)`

If Ultralytics isn’t installed, `train.py` will attempt to install it automatically.

## Setup

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

If you hit import errors during training (common on fresh environments), install these too:

```bash
pip install ultralytics albumentations
```

### 2) (Optional) sanity check

```bash
python test_environment.py
```

## Dataset expectations

Training expects a YOLO-style dataset produced by `prepare_dataset.py` under `dataset/`:

```
dataset/
   normal/
      train/images, train/labels
      val/images,   val/labels
   foggy/
      train/images, train/labels
      val/images,   val/labels
   rainy/
      train/images, train/labels
      val/images,   val/labels
```

If `dataset/` doesn’t exist, `run_pipeline.py` will call `prepare_dataset.py`.

## Run

### Easiest: full pipeline

```bash
python run_pipeline.py
```

### Step-by-step

```bash
python prepare_dataset.py
python train.py
python evaluate.py
python utils.py
```

## Outputs

Training artifacts are written under `runs/train/` (checkpoints, CSV/JSON summaries, plots).

## Notes

- Training performance depends heavily on GPU/CPU, dataset size, and the chosen YOLO backbone.
- If you only want to version-control code, keep `dataset/` and `runs/` ignored in Git.

## 🙏 Acknowledgments

- **Cityscapes**: Dataset providers (https://www.cityscapes-dataset.com/)
- **Domain Adaptation**: Based on DANN and other DA techniques

## 💡 Tips for Better Results

1. **Data Quality**: Ensure annotations are accurate
2. **Hyperparameter Tuning**: Experiment with learning rate and batch size
3. **Model Selection**: Try yolov11s for better accuracy (slower training)
4. **Data Augmentation**: Add more augmentation for robustness
5. **Longer Training**: Train for 100+ epochs for higher mAP

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [sakib.raihan@g.bracu.ac.bd]

---

**Happy Training! 🚀**
