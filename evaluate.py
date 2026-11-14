"""Validation evaluation for the solar filament segmentation pipeline.

Computes semantic Dice, semantic IoU, and instance-level Panoptic Quality (PQ)
on the validation split, for three ablation variants:
  A) No TTA, default thresholds
  B) 8-view TTA, default thresholds
  C) 8-view TTA, v12 submission thresholds

The validation split is reconstructed with the same seed and logic used in
train() so that the same images are evaluated.

Usage (Kaggle notebook or local with data available):
    python evaluate.py

Outputs a results table to stdout and writes evaluate_results.json.
"""
import json, random, cv2, numpy as np, torch
from scipy import ndimage as ndi

import filament as F

# ── Ablation configurations ──────────────────────────────────────────────────
VARIANTS = {
    "A_no_tta_default": dict(
        tta=False,
        t_lo=0.35, t_hi=0.99, min_meanp=0.85, min_area=50,
    ),
    "B_tta_default": dict(
        tta=True,
        t_lo=0.35, t_hi=0.99, min_meanp=0.85, min_area=50,
    ),
    "C_tta_v12": dict(
        tta=True,
        t_lo=0.35, t_hi=0.50, min_meanp=0.30, min_area=50,
    ),
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _predict_prob(model, img_512, tta):
    """Return (H,W) probability map at 512px."""
    x = torch.from_numpy(img_512.astype(np.float32))[None, None].to(F.dev)
    if tta:
        return F.tta_predict(model, x)          # already (512,512) numpy
    with torch.no_grad():
        return torch.sigmoid(model(x))[0, 0].cpu().numpy()


def _postprocess_cfg(prob_2048, cfg):
    """Apply per-component filtering with the given threshold config."""
    seed = (prob_2048 > cfg["t_lo"]).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (F.CFG["close_k"],) * 2)
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, k)
    lab, n = ndi.label(seed, structure=np.ones((3, 3)))
    masks = []
    for i in range(1, n + 1):
        m = lab == i
        r = prob_2048[m]
        if (r.max() > cfg["t_hi"]
                and r.mean() >= cfg["min_meanp"]
                and m.sum() >= cfg["min_area"]):
            masks.append(m)
    return masks


def semantic_dice_iou(pred_bin, gt_bin):
    """Pixel-level (semantic) Dice and IoU for one image."""
    inter = int((pred_bin & gt_bin).sum())
    pred_sum = int(pred_bin.sum())
    gt_sum = int(gt_bin.sum())
    union = pred_sum + gt_sum - inter
    # Both empty → perfect score
    if pred_sum == 0 and gt_sum == 0:
        return 1.0, 1.0
    dice = (2 * inter) / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0.0
    iou  = inter / union if union > 0 else 0.0
    return dice, iou


# ── Reconstruct validation split ─────────────────────────────────────────────
def get_val_items():
    random.seed(F.CFG["seed"])
    items = F.load_train_items()
    files = sorted({it[0] for it in items})
    random.shuffle(files)
    val_files = set(files[:max(1, int(F.CFG["val_frac"] * len(files)))])
    return [it for it in items if it[0] in val_files]


# ── Main evaluation loop ──────────────────────────────────────────────────────
def evaluate():
    # Load checkpoint
    model = F.build(pretrained=False).to(F.dev)
    model.load_state_dict(torch.load(F.CFG["ckpt"], map_location=F.dev))
    model.eval()

    val_items = get_val_items()
    print(f"Evaluating on {len(val_items)} validation items "
          f"({len({it[0] for it in val_items})} unique images)\n")

    O = F.CFG["out_size"]   # 2048
    results = {}

    for vname, vcfg in VARIANTS.items():
        dice_scores, iou_scores, pq_scores = [], [], []

        for fn, polys, h, w in val_items:
            # Ground truth at out_size
            gt_full = cv2.resize(
                F.poly_mask(polys, h, w), (O, O),
                interpolation=cv2.INTER_NEAREST,
            )

            # Probability map
            img = F.load_gray(F.os.path.join(F.CFG["train_dir"], fn))
            img_512 = F.small(img)
            prob_512 = _predict_prob(model, img_512, vcfg["tta"])
            prob_full = cv2.resize(prob_512, (O, O),
                                   interpolation=cv2.INTER_LINEAR)

            # Instance masks
            inst_masks = _postprocess_cfg(prob_full, vcfg)

            # Semantic binary prediction (union of all instances)
            pred_bin = np.zeros((O, O), dtype=bool)
            for m in inst_masks:
                pred_bin |= m
            gt_bin = gt_full.astype(bool)

            d, iou = semantic_dice_iou(pred_bin, gt_bin)
            pq = F.pq_score(inst_masks, gt_full)

            dice_scores.append(d)
            iou_scores.append(iou)
            pq_scores.append(pq)

        results[vname] = {
            "dice_mean":  float(np.mean(dice_scores)),
            "dice_std":   float(np.std(dice_scores)),
            "iou_mean":   float(np.mean(iou_scores)),
            "iou_std":    float(np.std(iou_scores)),
            "pq_mean":    float(np.mean(pq_scores)),
            "pq_std":     float(np.std(pq_scores)),
            "n_items":    len(val_items),
        }
        print(f"{vname}")
        print(f"  Dice  {results[vname]['dice_mean']:.4f} ± {results[vname]['dice_std']:.4f}")
        print(f"  IoU   {results[vname]['iou_mean']:.4f} ± {results[vname]['iou_std']:.4f}")
        print(f"  PQ    {results[vname]['pq_mean']:.4f} ± {results[vname]['pq_std']:.4f}\n")

    with open("evaluate_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results written to evaluate_results.json")
    return results


if __name__ == "__main__":
    evaluate()
