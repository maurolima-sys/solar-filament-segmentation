"""Solar filament segmentation: U-Net + postprocessing + RLE CSV.
Data: MAGFiLO_1.0_Kaggle_2026 (COCO-style polygons, 2048x2048 8-bit JPEG).
Usage: python filament.py (train + predict), or import and call predict().

Post-processing improvements over baseline (v2):
  - Per-component filtering: keep only if peak_prob > t_hi AND mean_prob >= min_meanp AND area >= min_area
  - TTA: 8 views (4 rotations x 2 flips), probabilities averaged before thresholding
"""
import os, glob, json, random, cv2, numpy as np, pandas as pd, torch, torch.nn as nn
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from scipy import ndimage as ndi
from pycocotools import mask as mu

# ---------------- CONFIG ----------------
ROOT = "/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"  # confirmed path
CFG = dict(
    ann_json=f"{ROOT}/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    train_dir=f"{ROOT}/train/train_images",
    test_dir=f"{ROOT}/test/test_images",
    size=512, bs=8, epochs=15, lr=3e-4, val_frac=0.2, seed=0,
    # categories: 1=Left, 2=Right, 3=Unidentifiable, 4=Ambiguous
    exclude_cats={4},          # Ambiguous excluded; set to set() to include all
    # post-processing (per-component) — tune with tune_postprocess() on your val set
    t_lo=0.35, t_hi=0.99, min_meanp=0.85, min_area=50,
    close_k=5, out_size=2048,
    encoder="resnet34", ckpt="/kaggle/working/best.pt",
)
dev = "cuda" if torch.cuda.is_available() else "cpu"
stem = lambda p: os.path.splitext(os.path.basename(p))[0]

def load_gray(p):
    x = cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    lo, hi = np.percentile(x, (1, 99))
    return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)

def small(img, interp=cv2.INTER_AREA):
    s = CFG["size"]
    return cv2.resize(img, (s, s), interpolation=interp)

def poly_mask(polys, h=2048, w=2048):
    if not polys: return np.zeros((h, w), np.uint8)
    return mu.decode(mu.merge(mu.frPyObjects(polys, h, w)))

def load_train_items():
    coco = json.load(open(CFG["ann_json"]))
    imgs = {im["id"]: im for im in coco["images"]}
    polys = defaultdict(list)
    for a in coco["annotations"]:
        if a["category_id"] not in CFG["exclude_cats"]:
            polys[a["image_id"]] += a["segmentation"]
    return [(imgs[i]["file_name"], polys[i], imgs[i]["height"], imgs[i]["width"]) for i in imgs]

class TrainDS(Dataset):
    def __init__(self, items, augment):
        self.augment, self.x, self.y = augment, [], []
        cache = {}
        for fn, polys, h, w in items:
            if fn not in cache:
                cache[fn] = small(load_gray(os.path.join(CFG["train_dir"], fn))).astype(np.float16)
            self.x.append(cache[fn])
            self.y.append(small(poly_mask(polys, h, w), cv2.INTER_NEAREST))
    def __len__(self): return len(self.x)
    def __getitem__(self, i):
        x, y = self.x[i].astype(np.float32), self.y[i].astype(np.float32)
        if self.augment:
            if random.random() < .5: x, y = x[:, ::-1], y[:, ::-1]
            if random.random() < .5: x, y = x[::-1], y[::-1]
            k = random.randint(0, 3); x, y = np.rot90(x, k), np.rot90(y, k)
        return torch.from_numpy(x.copy())[None], torch.from_numpy(y.copy())[None]

class TestDS(Dataset):
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, i): return torch.from_numpy(small(load_gray(self.paths[i])).astype(np.float32))[None]

def build(pretrained=True):
    return smp.Unet(CFG["encoder"], encoder_weights="imagenet" if pretrained else None, in_channels=1, classes=1)

def train():
    random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
    items = load_train_items()
    files = sorted({it[0] for it in items}); random.shuffle(files)
    val_files = set(files[:max(1, int(CFG["val_frac"] * len(files)))])
    tr = TrainDS([it for it in items if it[0] not in val_files], True)
    va = TrainDS([it for it in items if it[0] in val_files], False)
    print(len(tr), "train /", len(va), "val samples")
    nw = 2 if os.name != "nt" and dev == "cuda" else 0
    tl = DataLoader(tr, CFG["bs"], shuffle=True, num_workers=nw, pin_memory=dev=="cuda")
    vl = DataLoader(va, CFG["bs"], num_workers=nw, pin_memory=dev=="cuda")
    model = build().to(dev)
    opt = torch.optim.AdamW(model.parameters(), CFG["lr"])
    dice, bce = smp.losses.DiceLoss("binary"), nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    best = 1e9
    for ep in range(CFG["epochs"]):
        model.train()
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast("cuda", enabled=dev == "cuda"):
                o = model(x); loss = dice(o, y) + bce(o, y)
            if not torch.isfinite(loss): continue
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        model.eval(); vs = 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(dev), y.to(dev)
                with torch.autocast("cuda", enabled=dev == "cuda"):
                    o = model(x)
                v = (dice(o, y) + bce(o, y)).item()
                if torch.isfinite(torch.tensor(v)): vs += v * len(x)
        vs /= len(va); print(f"epoch {ep} val_loss {vs:.4f}")
        if vs < best:
            best = vs; torch.save(model.state_dict(), CFG["ckpt"])
    return model, [it for it in items if it[0] in val_files]

# ---- TTA helpers ----
_TTA = [
    (False, 0), (False, 1), (False, 2), (False, 3),
    (True,  0), (True,  1), (True,  2), (True,  3),
]

def _apply(x, flip, k):
    if flip: x = torch.flip(x, [-1])
    if k:    x = torch.rot90(x, k, [-2, -1])
    return x

def _undo(p, flip, k):
    if k:    p = torch.rot90(p, -k, [-2, -1])
    if flip: p = torch.flip(p, [-1])
    return p

def tta_predict(model, x):
    """x: (1,1,H,W) tensor -> averaged probability map (H,W) numpy."""
    acc = None
    for flip, k in _TTA:
        xi = _apply(x.clone(), flip, k)
        with torch.no_grad():
            p = torch.sigmoid(model(xi))
        p = _undo(p, flip, k)
        acc = p if acc is None else acc + p
    return (acc / len(_TTA))[0, 0].cpu().numpy()

# ---- Post-processing (per-component) ----
def postprocess(prob):
    """prob: HxW float at out_size -> list of boolean instance masks.

    A candidate component (seed from t_lo) is kept only if:
      peak_prob > t_hi  AND  mean_prob >= min_meanp  AND  area >= min_area
    This mirrors the approach from Dariush Bahrami's public notebook
    (solar-filament-u-net-oof-0-45-vs-public-0-39), adapted here without
    the gradient-boosting post-filter.
    """
    seed = (prob > CFG["t_lo"]).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CFG["close_k"],) * 2)
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, k)
    lab, n = ndi.label(seed, structure=np.ones((3, 3)))
    masks = []
    for i in range(1, n + 1):
        m = lab == i
        region = prob[m]
        if region.max() > CFG["t_hi"] and region.mean() >= CFG["min_meanp"] and m.sum() >= CFG["min_area"]:
            masks.append(m)
    return masks

def pq_score(pred_masks, gt_mask):
    """Panoptic Quality (PQ) for one image: list of bool pred masks vs binary gt.
    Matching threshold IoU >= 0.5.
    """
    gt_lab, n_gt = ndi.label(gt_mask, structure=np.ones((3, 3)))
    if not pred_masks and n_gt == 0: return 1.0
    if not pred_masks or n_gt == 0:  return 0.0
    tp = iou_sum = 0
    matched_gt = set()
    for pm in pred_masks:
        best_iou, best_g = 0.0, -1
        pm_area = pm.sum()
        for g in range(1, n_gt + 1):
            if g in matched_gt: continue
            gm = gt_lab == g
            inter = (pm & gm).sum()
            if inter == 0: continue
            iou = inter / (pm_area + gm.sum() - inter)
            if iou > best_iou: best_iou, best_g = iou, g
        if best_iou >= 0.5:
            tp += 1; iou_sum += best_iou; matched_gt.add(best_g)
    fp = len(pred_masks) - tp
    fn = n_gt - tp
    denom = tp + 0.5 * fp + 0.5 * fn
    return iou_sum / denom if denom else 0.0

def tune_postprocess(prob_maps, gt_masks):
    """Grid search post-processing thresholds on validation probability maps.

    Args:
        prob_maps: list of HxW float32 arrays (upsampled to out_size, one per val image)
        gt_masks:  list of HxW uint8 binary masks (ground truth, same size)

    Returns:
        best dict with keys t_lo, t_hi, min_meanp, min_area and its mean PQ.

    Usage after training:
        val_probs, val_gts = collect_val_probs(model, val_loader)
        best = tune_postprocess(val_probs, val_gts)
        CFG.update(best)
    """
    grid = dict(
        t_lo   =[0.25, 0.35, 0.45],
        t_hi   =[0.90, 0.95, 0.97, 0.99],
        min_meanp=[0.70, 0.80, 0.85],
        min_area =[0, 50, 200],
    )
    k_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CFG["close_k"],) * 2)
    best_pq, best_cfg = -1.0, {}
    from itertools import product
    combos = list(product(*grid.values()))
    print(f"grid search: {len(combos)} combinations over {len(prob_maps)} val images")
    for vals in combos:
        params = dict(zip(grid.keys(), vals))
        if params["t_lo"] >= params["t_hi"]: continue
        scores = []
        for prob, gt in zip(prob_maps, gt_masks):
            seed = (prob > params["t_lo"]).astype(np.uint8)
            seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, k_el)
            lab, n = ndi.label(seed, structure=np.ones((3, 3)))
            masks = []
            for i in range(1, n + 1):
                m = lab == i
                r = prob[m]
                if r.max() > params["t_hi"] and r.mean() >= params["min_meanp"] and m.sum() >= params["min_area"]:
                    masks.append(m)
            scores.append(pq_score(masks, gt))
        mean_pq = float(np.mean(scores))
        if mean_pq > best_pq:
            best_pq, best_cfg = mean_pq, params
    print(f"best val PQ {best_pq:.4f}: {best_cfg}")
    return best_cfg

def collect_val_probs(model, val_items):
    """Run TTA inference on val images; return (prob_maps, gt_masks) at out_size."""
    model.eval()
    O = CFG["out_size"]
    probs, gts = [], []
    for fn, polys, h, w in val_items:
        img = load_gray(os.path.join(CFG["train_dir"], fn))
        x = torch.from_numpy(small(img).astype(np.float32))[None, None].to(dev)
        pr = tta_predict(model, x)
        probs.append(cv2.resize(pr, (O, O), interpolation=cv2.INTER_LINEAR))
        gts.append(cv2.resize(poly_mask(polys, h, w), (O, O), interpolation=cv2.INTER_NEAREST))
    return probs, gts

def rle(mask):
    return mu.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("ascii")

def predict(out_csv="submission.csv"):
    model = build(pretrained=False).to(dev)
    model.load_state_dict(torch.load(CFG["ckpt"], map_location=dev))
    model.eval()
    paths = sorted(glob.glob(CFG["test_dir"] + "/*"))
    rows, O = [], CFG["out_size"]
    for p in paths:
        x = TestDS([p])[0][None, None].to(dev)          # (1,1,512,512)
        pr = tta_predict(model, x)                        # (512,512)
        pr = cv2.resize(pr, (O, O), interpolation=cv2.INTER_LINEAR)
        for j, inst in enumerate(postprocess(pr), 1):
            rows.append((f"{stem(p)}_{j}", rle(inst)))
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(out_csv, index=False)
    print(len(rows), "instances from", len(paths), "images saved to", out_csv)

if __name__ == "__main__":
    model, val_items = train()
    val_probs, val_gts = collect_val_probs(model, val_items)
    best = tune_postprocess(val_probs, val_gts)
    CFG.update(best)
    predict()
