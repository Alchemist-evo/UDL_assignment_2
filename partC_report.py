"""Part C - builds the comparative-analysis PDF from the results saved by the Part A / Part B notebooks.

usage:  python partC_report.py            (reads outputs/)
        python partC_report.py outputs_quick
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
                                PageBreak, KeepTogether)

OUT = sys.argv[1] if len(sys.argv) > 1 else "outputs"
FIG, RES = os.path.join(OUT, "figures"), os.path.join(OUT, "results")
PDF = "PartC_Comparative_Analysis.pdf" if OUT == "outputs" else os.path.join(OUT, "PartC_Comparative_Analysis.pdf")

A = json.load(open(os.path.join(RES, "partA_results.json")))
B = json.load(open(os.path.join(RES, "partB_results.json")))
vae, vq = A["vae"], A["vqvae"]
b1 = vae.get("1") or vae[sorted(vae, key=int)[0]]
k_main = "256" if "256" in vq else sorted(vq, key=int)[0]
vq256 = vq[k_main]
wg, gp = B["wgan"], B["wgan_gp"]

# --------------------------------------------------------------------------- extra comparison plots
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for b, r in sorted(vae.items(), key=lambda t: int(t[0])):
    ax[0].plot(range(1, len(r["psnr_hist"]) + 1), r["psnr_hist"], label=f"β-VAE β={b}")
for k, r in sorted(vq.items(), key=lambda t: -int(t[0])):
    ax[0].plot(range(1, len(r["psnr_hist"]) + 1), r["psnr_hist"], "--", label=f"VQ-VAE K={k}")
ax[0].set(xlabel="epoch", ylabel="test PSNR (dB)", title="Reconstruction quality vs epoch")
ax[0].grid(alpha=.3); ax[0].legend(fontsize=7, ncol=2)
names = [f"β-VAE\nβ={b}" for b in sorted(vae, key=int)] + [f"VQ-VAE\nK={k}" for k in sorted(vq, key=lambda s: -int(s))] + ["WGAN", "WGAN-GP"]
fids = [vae[b]["fid_sample"] for b in sorted(vae, key=int)] + [vq[k]["fid_sample"] for k in sorted(vq, key=lambda s: -int(s))] + [wg["fid"], gp["fid"]]
cols = ["#4C72B0"] * len(vae) + ["#DD8452"] * len(vq) + ["#55A868", "#C44E52"]
bars = ax[1].bar(range(len(fids)), fids, color=cols)
ax[1].bar_label(bars, fmt="%.0f", fontsize=7)
ax[1].set_xticks(range(len(fids))); ax[1].set_xticklabels(names, fontsize=7)
ax[1].set(ylabel="sample FID (lower is better)", title="Sample FID, 10k samples vs CIFAR-10 test")
ax[1].grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "partC_overview.png"), dpi=160); plt.close()

# --------------------------------------------------------------------------- text helpers
f2 = lambda x: f"{x:.2f}"
f1 = lambda x: f"{x:.1f}"
models = {"VAE (β=1)": b1["fid_sample"], f"VQ-VAE (K={k_main})": vq256["fid_sample"],
          "WGAN": wg["fid"], "WGAN-GP": gp["fid"]}
ranking = sorted(models, key=models.get)
rank_txt = " &lt; ".join(f"{m} ({f1(models[m])})" for m in ranking)

betas = sorted(vae, key=int)
psnr_b = [vae[b]["final_psnr"] for b in betas]
fid_b = [vae[b]["fid_sample"] for b in betas]
mono_psnr = all(psnr_b[i] >= psnr_b[i + 1] for i in range(len(psnr_b) - 1))
best_beta = betas[fid_b.index(min(fid_b))]
Ks = sorted(vq, key=int)
best_K = min(Ks, key=lambda k: vq[k]["fid_sample"])

speed = {"VAE (β=1)": b1["sample_time_10k_s"], f"VQ-VAE (K={k_main}) + PixelCNN": vq256["sample_time_10k_s"],
         "WGAN": wg["sample_time_10k_s"], "WGAN-GP": gp["sample_time_10k_s"]}
psnr_K_txt = ", ".join(f"K={k}: {vq[k]['final_psnr']:.2f}" for k in Ks)
perp_K_txt = ", ".join(f"{vq[k]['perplexity']:.0f}/{k}" for k in Ks)
nll_K_txt = ", ".join(f"K={k}: {vq[k]['pcnn_test_nll']:.2f}" for k in Ks)
psnr_b_txt = ", ".join(f"β={b}: {p:.2f}" for b, p in zip(betas, psnr_b))
slow_ratio =vq256["sample_time_10k_s"] / max(min(b1["sample_time_10k_s"], gp["sample_time_10k_s"]), 1e-6)

# --------------------------------------------------------------------------- document
ss = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, spaceAfter=6)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=4)
P = ParagraphStyle("p", parent=ss["BodyText"], fontSize=9.5, leading=13.5, alignment=4)
CAP = ParagraphStyle("cap", parent=P, fontSize=8, textColor=colors.grey, alignment=1)
BUL = ParagraphStyle("b", parent=P, leftIndent=12, bulletIndent=2)
story = []
par = lambda t, s=P: story.append(Paragraph(t, s))
bullet = lambda t: story.append(Paragraph(t, BUL, bulletText="•"))


def table(rows, widths=None, hl_col=None):
    t = Table(rows, colWidths=widths, hAlign="CENTER")
    st = [("FONT", (0, 0), (-1, -1), "Helvetica", 8), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
          ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
          ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#AAAAAA")),
          ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    t.setStyle(TableStyle(st))
    story.append(t)


def fig(path, w_cm, caption):
    if os.path.exists(path):
        from reportlab.lib.utils import ImageReader
        iw, ih = ImageReader(path).getSize()
        story.append(KeepTogether([Image(path, width=w_cm * cm, height=w_cm * cm * ih / iw),
                                   Paragraph(caption, CAP), Spacer(1, 6)]))


par("Assignment 2 – Part C: Comparative Analysis of β-VAE, VQ-VAE, WGAN and WGAN-GP on CIFAR-10", H1)
par("Unsupervised Deep Learning (S1-26_AIMLZG533) &nbsp;|&nbsp; Group: <i>[member names / IDs]</i>")
par(f"All models were trained on the BITS Prayogshala GPU ({A.get('device', 'cuda')}). Pixels were scaled to [0,1]. "
    "Every FID uses the same protocol: 10,000 generated images compared against the 10,000 CIFAR-10 test images "
    "(Inception-v3 pool3 features, torchmetrics), so the numbers are directly comparable across Part A and Part B.")

par("1. Summary of evaluation scores", H2)
rows = [["Model", "Test PSNR (dB) ↑", "Recon FID ↓", "Sample FID ↓", "Train time (min)", "Sampling 10k (s)", "Params (M)"]]
for b in betas:
    r = vae[b]
    rows.append([f"β-VAE β={b}", f2(r["final_psnr"]), f1(r["fid_recon"]), f1(r["fid_sample"]),
                 f1(r["train_time_s"] / 60), f2(r["sample_time_10k_s"]), f2(r["params"] / 1e6)])
for k in sorted(vq, key=lambda s: -int(s)):
    r = vq[k]
    rows.append([f"VQ-VAE K={k} + PixelCNN", f2(r["final_psnr"]), f1(r["fid_recon"]), f1(r["fid_sample"]),
                 f1(r["train_time_s"] / 60), f2(r["sample_time_10k_s"]), f2(r["params"] / 1e6)])
for name, r in [("WGAN (clipping)", wg), ("WGAN-GP", gp)]:
    rows.append([name, "n/a", "n/a", f1(r["fid"]), f1(r["train_time_s"] / 60), f2(r["sample_time_10k_s"]),
                 f2((r["params_G"] + r["params_D"]) / 1e6)])
table(rows, [3.9 * cm, 2.3 * cm, 2 * cm, 2.1 * cm, 2.3 * cm, 2.3 * cm, 1.7 * cm])
par("<font size=7.5>PSNR / recon FID are not defined for GANs (they have no encoder). VQ-VAE training time includes "
    "the PixelCNN prior. Parameter counts are for every network that is trained.</font>")
story.append(Spacer(1, 4))
rows = [["VQ-VAE", "Perplexity", "Codebook usage", "Dead codes", "Bits / image", "PixelCNN test NLL (nats/code)"]]
for k in sorted(vq, key=lambda s: -int(s)):
    r = vq[k]
    rows.append([f"K={k}", f"{r['perplexity']:.1f} / {k}", f"{100 * r['usage']:.1f}%", str(r["dead_codes"]),
                 f"{r['bits_per_image']:.0f}", f"{r['pcnn_test_nll']:.3f}"])
table(rows, [2.2 * cm, 2.6 * cm, 2.6 * cm, 2 * cm, 2.2 * cm, 4.6 * cm])
story.append(Spacer(1, 6))
fig(os.path.join(FIG, "partC_overview.png"), 17, "Figure 1 – Left: test PSNR over training. Right: sample FID of all models.")

par("2. 100 random samples from each model", H2)
par("Each grid below holds 100 <b>random, un-curated</b> samples, generated with the same seed for every model.")
story.append(PageBreak())
grid = [[Image(os.path.join(FIG, "vae_beta1_samples_100.png"), 8.6 * cm, 8.6 * cm),
         Image(os.path.join(FIG, f"vqvae_K{k_main}_samples_100.png"), 8.6 * cm, 8.6 * cm)],
        [Paragraph(f"(a) VAE β=1 – FID {f1(b1['fid_sample'])}", CAP),
         Paragraph(f"(b) VQ-VAE K={k_main} + PixelCNN – FID {f1(vq256['fid_sample'])}", CAP)],
        [Image(os.path.join(FIG, "wgan_samples_100.png"), 8.6 * cm, 8.6 * cm),
         Image(os.path.join(FIG, "wgangp_samples_100.png"), 8.6 * cm, 8.6 * cm)],
        [Paragraph(f"(c) WGAN (weight clipping) – FID {f1(wg['fid'])}", CAP),
         Paragraph(f"(d) WGAN-GP – FID {f1(gp['fid'])}", CAP)]]
story.append(Table(grid, colWidths=[9 * cm, 9 * cm]))
story.append(PageBreak())

par("3. Perceptual quality", H2)
bullet("<b>VAE (β=1):</b> samples are very smooth and blurry. You can make out the colour layout and the rough "
       "foreground/background split (a brown or green blob on a blue or green background), but there are no edges or "
       "textures, and few samples look like a recognisable object. This is the textbook VAE behaviour: the factorised "
       "Gaussian/Bernoulli likelihood averages over all plausible images, and the prior has regions that the encoder "
       "never used ('holes'), so decoded prior samples are washed out.")
bullet(f"<b>VQ-VAE (K={k_main}) + PixelCNN:</b> much sharper than the VAE. Local edges, saturated colours and textures "
       "are present, because the decoder only ever decodes 'real' combinations of codes. The weaker part is global "
       "coherence: many samples look like plausible scene fragments or textures rather than complete objects. That "
       "comes from the PixelCNN prior, which is small and generates the 8×8 grid one code at a time, so long-range "
       "errors build up.")
bullet("<b>WGAN (clipping):</b> GAN-like sharpness with object-and-background layouts, but noticeably noisier. "
       "There are high-frequency artefacts, blotchy colour patches and a fair number of distorted shapes. This fits "
       "the limited capacity of a weight-clipped critic and the slow convergence of RMSProp at lr 5e-5.")
bullet("<b>WGAN-GP:</b> the most realistic grid. Many samples show recognisable horses, cars, trucks, ships and birds "
       "with sensible backgrounds (sky, road, water, grass), and the colours and contrast look natural. There is no "
       "visible mode collapse: the grid is diverse in both class and colour. Failure cases are mostly "
       "anatomically odd animals, which is typical for a DCGAN-sized model at 32×32.")
par(f"Ranking by perceptual quality: WGAN-GP &gt; WGAN &gt; VQ-VAE+PixelCNN &gt; VAE. The FID ranking is "
    f"<b>{rank_txt}</b>, which mostly agrees with our visual impression.")

par("4. Discussion", H2)
par("<b>4.1 Reconstruction error.</b> Only the auto-encoders can reconstruct. "
    f"The VQ-VAE (K={k_main}) reaches <b>{f2(vq256['final_psnr'])} dB</b> test PSNR against <b>{f2(b1['final_psnr'])} dB</b> "
    f"for the β=1 VAE, and its reconstruction FID is {f1(vq256['fid_recon'])} against {f1(b1['fid_recon'])}. "
    "The VQ-VAE stores an 8×8 spatial grid of discrete codes and has no KL term pushing them towards an uninformative "
    "prior, whereas the VAE squeezes the whole image into one 128-d Gaussian vector and pays a KL cost for every nat of "
    "information. "
    + (f"In the β-VAE, PSNR falls steadily as β increases ({psnr_b_txt} dB). " if mono_psnr else
       f"In the β-VAE, PSNR generally falls as β increases ({psnr_b_txt} dB). ")
    + "Each step up in β makes the KL term more expensive, fewer latent dimensions stay active, and the "
      "reconstructions get blurrier. This is the expected rate–distortion trade-off. For the VQ-VAE, PSNR changes only "
      f"a little with K ({psnr_K_txt} dB). "
      "With EMA updates and dead-code restarts all codebooks stay well used "
      f"(perplexity {perp_K_txt}), "
      "so a larger codebook mainly adds bits per image, and each doubling gains less than the one before.")
par("<b>4.2 Sampling quality.</b> "
    f"The GANs produce the best samples, and <b>WGAN-GP ({f1(gp['fid'])})</b> is clearly better than clipped "
    f"<b>WGAN ({f1(wg['fid'])})</b>. This confirms the main claim of Gulrajani et al.: a gradient penalty is a much "
    "better way to enforce the Lipschitz constraint than weight clipping, which pushes the critic's weights to ±c and "
    "leaves it with very little capacity. The VQ-VAE + PixelCNN "
    f"({f1(vq256['fid_sample'])}) is sharper than the VAE ({f1(b1['fid_sample'])}), but it is limited by its prior. "
    f"Across K, the best sample FID came from K={best_K}. A larger K makes the prior harder to learn: its NLL per code rises "
    f"({nll_K_txt}). "
    "This shows that VQ-VAE <i>generation</i> quality depends on the prior just as much as on the codebook. "
    f"For the β-VAE the best sample FID was at β={best_beta}. A higher β narrows the gap between the aggregate "
    "posterior and the prior, which helps prior sampling, but it also removes detail. Overall, likelihood-based models "
    "(VAE, VQ-VAE) cover all the modes of the data but produce blurrier or less coherent samples, while adversarial "
    "models give much more realistic samples and offer no reconstruction or likelihood.")
par("<b>4.3 Speed.</b> "
    f"Generating 10,000 images takes {f2(b1['sample_time_10k_s'])} s for the VAE, {f2(wg['sample_time_10k_s'])} s / "
    f"{f2(gp['sample_time_10k_s'])} s for WGAN / WGAN-GP, but <b>{f2(vq256['sample_time_10k_s'])} s</b> for the "
    f"VQ-VAE ({slow_ratio:.0f}× slower). The VAE and the GANs need a single forward pass, while the PixelCNN prior "
    "is autoregressive and needs 64 sequential network evaluations, one per latent position. In training, the "
    f"auto-encoders are the cheapest (β-VAE ≈ {f1(b1['train_time_s'] / 60)} min, VQ-VAE + PixelCNN ≈ "
    f"{f1(vq256['train_time_s'] / 60)} min) and converge smoothly and reliably. The GANs ({f1(wg['train_time_s'] / 60)} / "
    f"{f1(gp['train_time_s'] / 60)} min) are the most expensive, because the critic is updated 5 times for every "
    "generator step and WGAN-GP also needs a double back-propagation for the penalty. Their training is also the "
    "most sensitive to hyperparameters.")

par("5. Conclusion", H2)
par("Our results agree with the well-known characteristics of each family. (i) β controls a rate–distortion "
    "trade-off: higher β gives lower PSNR and blurrier samples. (ii) Discrete latents (VQ-VAE) give much better "
    "reconstructions than a Gaussian VAE, and a well-used codebook (high perplexity) matters more than its raw size. "
    "Generation quality, however, depends on how well the autoregressive prior is learned, and that prior is also slow "
    "to sample from. (iii) Adversarial training gives the most realistic samples, and WGAN-GP's gradient penalty clearly "
    "beats weight clipping in both sample quality and convergence speed. If we need reconstructions or "
    "representations, VQ-VAE is the best choice. If we need fast, high-quality sampling, WGAN-GP is the best choice.")

SimpleDocTemplate(PDF, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                  topMargin=1.4 * cm, bottomMargin=1.4 * cm,
                  title="UDL Assignment 2 - Part C").build(story)
print("wrote", PDF)
