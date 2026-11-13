# Solar Filament Segmentation 2026

U-Net (ResNet34 encoder, 1-channel input) trained on MAGFiLO H-alpha images.

## Pipeline (v2)

1. **Preprocess**: percentile (1–99) normalisation, resize 2048 → 512 (grayscale).
   Masks rasterised from COCO polygons; 80/20 train/val split by image file (no annotator leakage).
2. **Model**: `segmentation_models_pytorch.Unet(resnet34)`, Dice + BCE loss, AdamW, fp16 training on GPU.
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

The per-component filtering strategy (t_lo / t_hi / min_meanp) was inspired by
Dariush Bahrami's public notebook *"Solar Filament U-Net OOF 0.45 vs Public 0.39"*
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

## Files

- `filament.py` — dataset, training, TTA, post-processing, RLE, prediction
- `pipeline.ipynb` — end-to-end inference notebook
- `requirements.txt` — Python dependencies
- `weights/best.pt` — trained weights (float32); if > 100 MB use Git LFS or link to Kaggle dataset
- `reference/` — third-party notebooks kept for reference, **not submitted**
