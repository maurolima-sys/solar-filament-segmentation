"""
make_figures.py — gera as figuras do relatório a partir de imagens reais de validação.

Uso (após treino):
    python make_figures.py

Saída: report/fig1_sample.png, report/fig2_comparison.png, report/fig3_hard.png
Requer: weights/best.pt e acesso ao dataset de treino (CFG["train_dir"] + CFG["ann_json"]).
"""
import os, random, cv2, numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import ndimage as ndi
import filament as F

OUT = "report"
os.makedirs(OUT, exist_ok=True)

# ── carrega modelo ────────────────────────────────────────────────────────────
model = F.build(pretrained=False).to(F.dev)
model.load_state_dict(torch.load(F.CFG["ckpt"], map_location=F.dev))
model.eval()

# ── reconstrói split de validação (mesma seed do treino) ─────────────────────
random.seed(F.CFG["seed"])
items = F.load_train_items()
files = sorted({it[0] for it in items})
random.shuffle(files)
val_files = set(files[:max(1, int(F.CFG["val_frac"] * len(files)))])
val_items = [it for it in items if it[0] in val_files]

# ── coleta probabilidades e GT para todos os itens de val ────────────────────
print(f"Collecting probs for {len(val_items)} val items...")
probs, gts, fnames = [], [], []
seen = {}
for fn, polys, h, w in val_items:
    if fn in seen:
        continue
    seen[fn] = True
    img = F.load_gray(os.path.join(F.CFG["train_dir"], fn))
    x = torch.from_numpy(F.small(img).astype(np.float32))[None, None].to(F.dev)
    pr = F.tta_predict(model, x)
    O = F.CFG["out_size"]
    pr_up = cv2.resize(pr, (O, O), interpolation=cv2.INTER_LINEAR)
    gt_up = cv2.resize(F.poly_mask(polys, h, w), (O, O), interpolation=cv2.INTER_NEAREST)
    probs.append(pr_up); gts.append(gt_up); fnames.append(fn)

# ── seleciona amostras representativas ───────────────────────────────────────
def _n_filaments(gt):
    _, n = ndi.label(gt, structure=np.ones((3, 3)))
    return n

# amostra com filamentos (para fig1 e fig2)
with_fil = [(p, g, f) for p, g, f in zip(probs, gts, fnames) if _n_filaments(g) >= 2]
random.seed(42)
sample = random.choice(with_fil) if with_fil else (probs[0], gts[0], fnames[0])

# casos difíceis: muitos filamentos ou filamentos pequenos
hard = sorted(zip(probs, gts, fnames), key=lambda x: _n_filaments(x[1]), reverse=True)[:3]

# ── helpers ───────────────────────────────────────────────────────────────────
def _load_orig(fn):
    img = F.load_gray(os.path.join(F.CFG["train_dir"], fn))
    return cv2.resize(img, (F.CFG["out_size"], F.CFG["out_size"]), interpolation=cv2.INTER_AREA)

def _pred_mask(prob):
    F.CFG  # usa thresholds actuais
    masks = F.postprocess(prob)
    m = np.zeros(prob.shape, np.uint8)
    for inst in masks:
        m[inst] = 1
    return m

def _error_map(gt, pred):
    tp = (gt > 0) & (pred > 0)
    fp = (gt == 0) & (pred > 0)
    fn = (gt > 0) & (pred == 0)
    rgb = np.zeros((*gt.shape, 3), np.uint8)
    rgb[tp] = [255, 255, 255]   # branco — acerto
    rgb[fp] = [255,  80,  80]   # vermelho — falso positivo
    rgb[fn] = [80,  80, 255]    # azul — falso negativo
    return rgb

# ── Figura 1: imagem original + GT ───────────────────────────────────────────
prob1, gt1, fn1 = sample
orig1 = _load_orig(fn1)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(orig1, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("H-alpha (original)", fontsize=9)
axes[0].axis("off")

axes[1].imshow(orig1, cmap="gray", vmin=0, vmax=1)
axes[1].imshow(gt1, cmap="Reds", alpha=0.45, vmin=0, vmax=1)
axes[1].set_title("Ground Truth (expert)", fontsize=9)
axes[1].axis("off")

fig.suptitle(f"Sample: {fn1}", fontsize=7, y=0.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_sample.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("fig1_sample.png saved")

# ── Figura 2: original / GT / prediction / error map ─────────────────────────
pred1 = _pred_mask(prob1)
err1  = _error_map(gt1, pred1)

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
titles = ["(a) Original", "(b) Ground Truth", "(c) Prediction", "(d) Error map"]
imgs   = [
    (orig1, "gray"),
    (gt1,   "Reds"),
    (pred1, "Blues"),
    (err1,  None),
]
for ax, (im, cmap), title in zip(axes, imgs, titles):
    if cmap:
        ax.imshow(orig1, cmap="gray", vmin=0, vmax=1)
        if cmap != "gray":
            ax.imshow(im, cmap=cmap, alpha=0.5, vmin=0, vmax=1)
    else:
        ax.imshow(im)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

patches = [
    mpatches.Patch(color=(1, 1, 1), label="TP"),
    mpatches.Patch(color=(1, .31, .31), label="FP"),
    mpatches.Patch(color=(.31, .31, 1), label="FN"),
]
axes[3].legend(handles=patches, loc="lower right", fontsize=7, framealpha=0.7)
fig.suptitle(f"Validation sample: {fn1}", fontsize=7, y=0.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_comparison.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("fig2_comparison.png saved")

# ── Figura 3: três casos difíceis ────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
for row, (ph, gh, fh) in enumerate(hard):
    oh = _load_orig(fh)
    predh = _pred_mask(ph)
    errh  = _error_map(gh, predh)
    n_gt  = _n_filaments(gh)
    n_pr  = _n_filaments(predh)

    axes[row, 0].imshow(oh, cmap="gray", vmin=0, vmax=1)
    axes[row, 0].set_title(f"Original\n{fh[:20]}", fontsize=7)
    axes[row, 0].axis("off")

    axes[row, 1].imshow(oh, cmap="gray", vmin=0, vmax=1)
    axes[row, 1].imshow(gh, cmap="Reds", alpha=0.5, vmin=0, vmax=1)
    axes[row, 1].set_title(f"GT ({n_gt} filaments)", fontsize=7)
    axes[row, 1].axis("off")

    axes[row, 2].imshow(errh)
    axes[row, 2].set_title(f"Error (pred {n_pr})", fontsize=7)
    axes[row, 2].axis("off")

fig.suptitle("Hard cases — images with most filaments", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_hard.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("fig3_hard.png saved")
print("Done. Files in report/")
