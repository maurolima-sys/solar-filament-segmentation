# Handoff — Solar Filament Segmentation 2026

**Data:** 14 Nov 2026  
**Prazo final:** 15 Nov 2026, 23:59 CDT (= 16 Nov 2026, 01:59 BRT)

---

## Estado atual

| Item | Estado | Detalhe |
|---|---|---|
| Submissão CSV no Kaggle | ✅ | ref `58326011`, 960 instâncias, 180 imagens |
| Score público | ⏳ pendente | verificar em https://www.kaggle.com/competitions/filament-segmentation-2026/leaderboard |
| `weights/best.pt` no GitHub | ✅ | commit `0ac38dc`, 93 MB |
| `pipeline.ipynb` no GitHub | ✅ | carrega `best.pt`, executa inferência completa |
| `requirements.txt` no GitHub | ✅ | |
| `check_submission.py` | ✅ | monitoriza score via API Kaggle |
| Relatório técnico (PDF) | ❌ **FALTA** | `report/report.tex` existe, falta compilar e submeter |
| Google Form submetido | ❌ **FALTA** | obrigatório para avaliação pelo júri |

---

## Tarefas para amanhã (por ordem de prioridade)

### 1. Verificar o score público
```powershell
python check_submission.py
```
Ou acede directamente: https://www.kaggle.com/competitions/filament-segmentation-2026/leaderboard

### 2. Compilar o relatório técnico
O ficheiro `report/report.tex` foi descarregado do Kaggle para `.qodo/weights_download/repo/report/report.tex`.

Opções para compilar:
- **Overleaf** (recomendado): faz upload do `.tex` em https://www.overleaf.com → compila → descarrega PDF
- **Local** (se tiveres LaTeX instalado):
  ```powershell
  cd report
  pdflatex report.tex
  ```

O template exigido pela competição está em:
https://www.overleaf.com/latex/templates/filament-segmentation-2026/  
(ver secção "Final Submission" na página da competição)

O relatório deve ter **4 páginas** e cobrir:
- Pré-processamento (normalização percentil 1–99, resize 512)
- Arquitectura (U-Net ResNet34, 1 canal, imagenet weights)
- Treino (15 épocas, AdamW, Dice+BCE, fp16, 80/20 split, excluir cat 4)
- TTA (8 vistas: 4 rotações × 2 flips)
- Pós-processamento (t_lo=0.35, t_hi=0.50, min_meanp=0.30, min_area=50, closing k=5)
- Grid search nos thresholds no val set
- Resultado: 960 instâncias em 180 imagens de teste

### 3. Submeter o Google Form
Link do form: ver secção "Final Submission" em https://www.kaggle.com/competitions/filament-segmentation-2026/overview

O form exige:
- PDF do relatório técnico (4 páginas)
- Link do repositório público: https://github.com/maurolima-sys/solar-filament-segmentation

### 4. Seleccionar submissão final no Kaggle
A competição permite seleccionar **até 2 submissões finais**.  
Acede a: https://www.kaggle.com/competitions/filament-segmentation-2026/submissions  
Selecciona a ref `58326011` como submissão final.

---

## Repositório GitHub

URL: https://github.com/maurolima-sys/solar-filament-segmentation  
HEAD: `0ac38dc`

```
filament.py          — pipeline completo
pipeline.ipynb       — notebook de inferência
weights/best.pt      — pesos v12 (93 MB)
requirements.txt     — dependências
check_submission.py  — monitoriza score Kaggle
report/report.tex    — relatório (compilar → PDF)
```

## Parâmetros v12 (para o relatório)

```python
CFG = dict(
    size=512, bs=8, epochs=15, lr=3e-4, val_frac=0.2, seed=0,
    exclude_cats={4},           # Ambiguous excluído
    t_lo=0.35, t_hi=0.50,
    min_meanp=0.30, min_area=50,
    close_k=5, out_size=2048,
    encoder="resnet34",
)
```

TTA: 8 vistas (4 rotações × 2 flips), probabilidades médias antes do threshold.  
Commit Kaggle executado: `0e6f403`  
Log confirma: `960 instances from 180 images saved to /kaggle/working/submission.csv`

---

## Segurança — ACÇÃO PENDENTE

Os tokens abaixo foram expostos acidentalmente nesta sessão e **devem ser revogados**:

- **Kaggle API token**: https://www.kaggle.com/settings → API → Expire Token → gerar novo `kaggle.json`
- **GitHub PAT**: https://github.com/settings/tokens → Delete → gerar novo token

Após rotacionar, actualizar o `.env` local (nunca commitar o `.env`).
