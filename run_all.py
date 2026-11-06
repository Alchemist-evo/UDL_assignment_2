"""Runs the whole assignment inside the venv: executes both notebooks (outputs embedded),
exports them to HTML and builds the Part C PDF.

usage:  python run_all.py            full run
        python run_all.py --quick    tiny smoke test (writes to outputs_quick/, notebooks copied there)
        python run_all.py --only A   just Part A  (A | B | C)
"""
import argparse, os, shutil, subprocess, sys, time
import nbformat
from nbclient import NotebookClient

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--only", choices=["A", "B", "C"])
args = ap.parse_args()

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
os.environ["UDL_QUICK"] = "1" if args.quick else "0"
out_dir = "outputs_quick" if args.quick else "outputs"
os.makedirs(out_dir, exist_ok=True)

NOTEBOOKS = {"A": "PartA_BetaVAE_VQVAE.ipynb", "B": "PartB_WGAN_WGANGP.ipynb"}


def run_notebook(name):
    src = os.path.join(ROOT, name)
    dst = os.path.join(out_dir, name) if args.quick else src     # quick runs never touch the real notebooks
    nb = nbformat.read(src, as_version=4)
    print(f">> executing {name} ...", flush=True)
    t = time.time()
    NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": ROOT}}).execute()
    nbformat.write(nb, dst)
    subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "html", dst], check=True)
    print(f">> {name} done in {(time.time() - t) / 60:.1f} min -> {dst} + .html", flush=True)


for part in ["A", "B"]:
    if args.only in (None, part):
        run_notebook(NOTEBOOKS[part])
if args.only in (None, "C"):
    subprocess.run([sys.executable, "partC_report.py", out_dir], check=True)
# PDF copies of the executed notebooks (printed from the HTML exports)
subprocess.run([sys.executable, "notebook_pdf.py"] + ([out_dir] if args.quick else []), check=True)
