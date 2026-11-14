# Solar Filament Segmentation 2026

U-Net (ResNet34 encoder, 1-channel input) trained on MAGFiLO H-alpha images.
Kaggle competition: [filament-segmentation-2026](https://www.kaggle.com/competitions/filament-segmentation-2026).

## Current status (2026-11-14)

The v13 submission (`submission_v13.csv`) received a **public score of 0.28**.
It contains 1,790 predicted instances from 178 of the 180 test images; the
other two have no predicted instances in the CSV. The previous v12 submission
scored 0.00.
The SHA-256 of the matching `weights/best.pt` checkpoint is
`85f9103a6423875996a3f0200495ea862ed8e945b88305676728b3e127c709f5`.
The SHA-256 of `submission_v13.csv` is
`b87bddccc8121123948ffcae40e1d0a00a809f0f1bf196a2113f9be2faac68a8`.

The v13 checkpoint reached validation Dice 0.6275 at epoch 33. This Dice was
computed at a 0.5 probability threshold on randomly sampled 512×512 crops from
1024×1024 resized validation images; it is not a full-image PQ measurement.
The competition score is the authoritative public result.

An earlier CPU sweep reported local PQ 0.0000 and semantic Dice 0.1144. That
sweep loaded the older `best.pt` checkpoint from Kaggle input, not v13's
`best_v13.pt`, so it does not evaluate the v13 model and is not comparable.
Its PQ is also a local connected-component approximation, not the official
Kaggle evaluator.

## Pipeline (v13)

1. **Preprocess**: convert to grayscale, normalize between the 1st and 99th
   percentiles, and resize 2048×2048 images to 1024×1024. Training masks are
   rasterised from COCO polygons; the 80/20 split is by image file. Category 4
   (Ambiguous) is excluded.
2. **Model and training**: one-channel U-Net with a ResNet34 encoder. The v13
   run used Dice loss plus binary cross-entropy with `pos_weight=20`, AdamW
   (`lr=5e-4`), 35 epochs, and 512×512 random crops from the resized images.
   The learning rate used a three-epoch warmup followed by cosine decay. The
   checkpoint was selected by validation Dice.
3. **Inference**: process the full 1024×1024 resized image without test-time
   augmentation (TTA), then upsample the probability map to 2048×2048.
4. **Post-processing**: threshold seeds at `t_lo=0.35`, apply a 5×5
   morphological closing, and retain connected components meeting
   `t_hi=0.50`, `min_meanp=0.30`, and `min_area=50`.
5. **Output**: encode each retained component as one RLE row in
   `submission_v13.csv`.

The earlier v12 pipeline used 512×512 inputs and 8-view TTA. The v13 inference
settings above match the checkpoint and submission that received the public
score of 0.28.

The per-component filtering strategy (t_lo / t_hi / min_meanp) was inspired by
Dariush Afshar's public notebook *"Solar Filament U-Net OOF 0.45 vs Public 0.39"*
(Kaggle, 2026). The gradient-boosting post-filter described there was not adopted.
The notebook is kept locally in `reference/` for consultation only and is **not**
part of this submission.

## Reproduce v13 (inference only)

```bash
pip install -r requirements.txt
jupyter notebook pipeline.ipynb
```

Attach the MAGFiLO dataset and set `filament.CFG` data paths if they are not
discovered automatically. The notebook loads `weights/best.pt`, uses the fixed
v13 configuration above, does not retrain or retune thresholds, and writes
`submission_v13.csv`. Inference runs on CPU or GPU; a GPU is recommended.

The `filament.py` training function is the earlier 512-pixel training pipeline;
it does not reproduce the v13 crop-based training run. The included checkpoint
and inference notebook reproduce the v13 predictions without retraining.

## Files

- `filament.py` — shared dataset/model code, earlier 512-pixel training, TTA,
  post-processing, RLE, and prediction
- `pipeline.ipynb` — v13 inference and submission validation notebook
- `report/report.tex` — technical report source
- `report/report.pdf` — compiled four-page technical report
- `requirements.txt` — Python dependencies
- `weights/best.pt` — exact v13 checkpoint (SHA-256 listed above)
- `submission_v13.csv` — frozen CSV that received the public score of 0.28
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
