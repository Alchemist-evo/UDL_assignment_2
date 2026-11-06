"""Embeds Prayogshala_lab_screenshot.jpg into the 'Hardware / lab infrastructure' markdown cell of both notebooks
(as a cell attachment, so it also appears in the HTML/PDF exports). Safe to run more than once, and works on
executed notebooks without touching their outputs.

usage:  python embed_lab_screenshot.py [notebook.ipynb ...]      (default: both Part A / Part B notebooks)
"""
import base64, io, os, sys
import nbformat
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
SHOT = os.path.join(ROOT, "Prayogshala_lab_screenshot.jpg")
NAME = "prayogshala_lab.jpg"
MARK = "### Hardware / lab infrastructure"

im = Image.open(SHOT).convert("RGB")
im.thumbnail((1600, 1600))                                   # keep the notebook small
buf = io.BytesIO(); im.save(buf, "JPEG", quality=85)
data = base64.b64encode(buf.getvalue()).decode()

nbs = sys.argv[1:] or [os.path.join(ROOT, n) for n in ("PartA_BetaVAE_VQVAE.ipynb", "PartB_WGAN_WGANGP.ipynb")]
for path in nbs:
    nb = nbformat.read(path, as_version=4)
    cell = next(c for c in nb.cells if c.cell_type == "markdown" and c.source.startswith(MARK))
    cell["attachments"] = {NAME: {"image/jpeg": data}}
    if f"attachment:{NAME}" not in cell.source:
        cell.source += f"\n\n![BITS Prayogshala lab session](attachment:{NAME})\n\n*BITS Prayogshala remote desktop with this assignment's code in the lab workspace.*"
    nbformat.write(nb, path)
    print("embedded screenshot in", path)
