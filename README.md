# UDL Assignment 2: Image generation with β-VAE, VQ-VAE + PixelCNN, WGAN and WGAN-GP (CIFAR-10)

| File | What it is |
|---|---|
| `PartA_BetaVAE_VQVAE.ipynb` | Part A: β-VAE (β ∈ {1,2,4,10}) and VQ-VAE (K ∈ {512,256,128}) with a Gated PixelCNN prior |
| `PartB_WGAN_WGANGP.ipynb` | Part B: WGAN (weight clipping) and WGAN-GP |
| `partC_report.py` | Part C: builds `PartC_Comparative_Analysis.pdf` from the saved results and figures |
| `run_all.py` | Runs A → B → C end to end (executes the notebooks, embeds the outputs, exports HTML + PDF) |
| `embed_lab_screenshot.py` | Embeds the Prayogshala screenshot into both notebooks (safe to re-run on executed notebooks) |
| `notebook_pdf.py` | Prints the executed notebooks' HTML to PDF with headless Chrome/Edge (no LaTeX needed) |
| `setup_env.sh` / `setup_env.ps1` | Create `.venv` on Linux (Fedora/Ubuntu) / Windows |

## 1. Environment
The code automatically uses an **NVIDIA GPU (CUDA)**, then an **Intel Arc GPU (XPU)**, and falls back to the CPU.

**Fedora / Linux (and the BITS lab machine)**
```bash
sudo dnf install -y python3 python3-pip          # Fedora (Ubuntu: apt install python3-venv)
# Intel Arc on Fedora only, GPU compute runtime:
# sudo dnf install -y intel-compute-runtime oneapi-level-zero
bash setup_env.sh                                  # auto-detects cuda / xpu / cpu
source .venv/bin/activate
```
**Windows**
```powershell
powershell -ExecutionPolicy Bypass -File setup_env.ps1
.\.venv\Scripts\Activate.ps1
```

## 2. Run
```bash
python run_all.py --quick     # ~5 min smoke test, results in outputs_quick/ (not for submission)
python run_all.py             # full run: notebooks executed in place + .html + PartC pdf
python run_all.py --only B    # run a single part (A | B | C)
```
Each model saves a checkpoint to `outputs/checkpoints/`. If a session dies, re-running skips the models that are already trained.
You can also open the notebooks in Jupyter and run them cell by cell (kernel "Python (udl-a2)").

Rough full-run time: Part A ≈ 1.5 h and Part B ≈ 2–2.5 h, measured on an Intel Arc A770 (a lab NVIDIA GPU should be similar or faster). The epoch and iteration counts are in the `CFG` cell of each notebook.

## 3. Submission checklist
- [ ] `PartA_BetaVAE_VQVAE.ipynb` + `.html` + `.pdf` (outputs embedded)
- [ ] `PartB_WGAN_WGANGP.ipynb` + `.html` + `.pdf`
- [ ] `PartC_Comparative_Analysis.pdf` (lab screenshot in the appendix)
- [ ] `Prayogshala_lab_screenshot.jpg` (also embedded in both notebooks via `python embed_lab_screenshot.py`)

## 4. Group 93
| # | Name | BITS ID | Contribution |
|---|---|---|---|
| 1 | Prajwal Shetty K P | 2025AE05434 | Part B: WGAN & WGAN-GP |
| 2 | Tanushi Agarwal | 2025AE05410 | Part A: β-VAE, VQ-VAE + PixelCNN |
| 3 | Shahabuddin | 2025AE05328 | Data pipeline, PSNR/FID evaluation, lab GPU runs |
| 4 | SANTHOSH M | 2025AE05426 | Part C: comparative analysis & PDF |
| 5 | TUFAIL AHMAD | 2025AE05400 | Visualisation, result tables, HTML export & review |
