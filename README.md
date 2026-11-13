# Solar Filament Segmentation 2026

U-Net (ResNet34 encoder, 1-channel input) trained on MAGFiLO H-alpha images.
Kaggle competition: [filament-segmentation-2026](https://www.kaggle.com/competitions/filament-segmentation-2026).

## Pipeline (v2)

1. **Preprocess**: percentile (1–99) normalisation, resize 2048 → 512 (grayscale).
   Masks rasterised from COCO polygons; 80/20 train/val split by image file (no annotator leakage).
   Category 4 (Ambiguous) excluded from training.
2. **Model**: `segmentation_models_pytorch.Unet(resnet34, encoder_weights="imagenet")`, Dice + BCE loss, AdamW, fp16 training on GPU.
3. **TTA**: 8 views (4 rotations × 2 flips), probabilities averaged before thresholding.
4. **Post-process**: upsample to 2048×2048; seed mask at `t_lo` + morphological closing;
   keep component only if `peak_prob > t_hi`, `mean_prob ≥ min_meanp`, `area ≥ min_area`.
   Thresholds tuned by grid search on the validation set after training.
5. **Output**: one RLE per filament → `submission.csv`.

### Comparison with baseline

| | Baseline (v1) | This repo (v2) |
|---|---|---|
| Resolution | 512 | 512 |
| TTA | none | 8 views |
| Post-processing | single threshold + min_area | per-component peak/mean/area filter + grid search |
| Exclude ambiguous | no | yes (cat 4) |

The per-component filtering strategy (t_lo / t_hi / min_meanp) was inspired by
Dariush Afshar's public notebook *"Solar Filament U-Net OOF 0.45 vs Public 0.39"*
(Kaggle, 2026). The gradient-boosting post-filter described there was not adopted.
The notebook is kept locally in `reference/` for consultation only and is **not**
part of this submission.

## Reproduce (inference only)

```bash
pip install -r requirements.txt
jupyter notebook pipeline.ipynb   # loads weights/best.pt, writes submission.csv
```

Set `CFG["test_dir"]` to the folder with the test JPEGs. Runs on CPU or GPU.

## Retrain

```bash
python filament.py   # set CFG paths first; runs train → grid search → predict
```

Or on Kaggle (clones this repo and runs the full pipeline):

```python
!pip install -q segmentation-models-pytorch pycocotools
!git clone --depth 1 https://github.com/maurolima-sys/solar-filament-segmentation.git /kaggle/working/repo
import sys; sys.path.insert(0, '/kaggle/working/repo')
import filament as F
F.train()
```

## Files

- `filament.py` — dataset, training, TTA, post-processing, RLE, prediction
- `pipeline.ipynb` — end-to-end inference notebook
- `requirements.txt` — Python dependencies
- `weights/best.pt` — trained weights (float32); if > 100 MB use Git LFS or link to Kaggle dataset
- `reference/` — third-party notebooks kept for reference, **not submitted**

## External benchmarks

| Notebook | Approach | Public LB |
|---|---|---|
| `lamhuy8904/solar-filament-unet-segmentation-0-55` | U-Net | 0.55+ |
| `dariushafshar/solar-filament-u-net-oof-0-45-vs-public-0-39` | U-Net × 10, 2048px | 0.39 pub / 0.45 OOF |
| `phuongncn/lb-0-70-topology-safe-solar-filaments-handoff` | Topology-safe | 0.70 |
| `timgluhkikh/best-score-magfilo-yolo11s-detect-crop-u-net` | YOLO11s → crop → U-Net | best score |

Scores are not directly comparable until metric, dataset split and submission protocol are verified.

## Experimental policy

Do not modify the pipeline based solely on external leaderboard scores.
Establish a reproducible baseline first, then compare under documented evaluation conditions.
