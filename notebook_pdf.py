"""Converts the executed notebooks' HTML exports to PDF with headless Chrome/Edge (no LaTeX needed).

usage:  python notebook_pdf.py                       (PartA/PartB .html in the repo root)
        python notebook_pdf.py outputs_quick         (HTML files in another folder)
"""
import os, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, sys.argv[1]) if len(sys.argv) > 1 else ROOT
NAMES = ["PartA_BetaVAE_VQVAE", "PartB_WGAN_WGANGP"]
BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            "google-chrome", "chromium", "chromium-browser", "microsoft-edge"]
# wrap long code lines / outputs instead of cutting them off at the page edge
PRINT_CSS = """<style>@media print {
  pre, .highlight pre, .jp-OutputArea-output pre { white-space: pre-wrap !important; word-break: break-word; }
  .jp-Cell, .jp-OutputArea-output img { break-inside: avoid-page; }
  body { font-size: 11px; }
}</style>"""

browser = next((b for b in BROWSERS if shutil.which(b) or os.path.exists(b)), None)
if browser is None:
    sys.exit("No Chrome/Edge/Chromium found for printing to PDF")

for name in NAMES:
    html = os.path.join(SRC, name + ".html")
    if not os.path.exists(html):
        print("skip (missing):", html); continue
    page = open(html, encoding="utf-8").read().replace("</head>", PRINT_CSS + "</head>", 1)
    tmp = os.path.join(tempfile.mkdtemp(), name + ".html")
    open(tmp, "w", encoding="utf-8").write(page)
    pdf = os.path.join(SRC, name + ".pdf")
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=30000",            # let MathJax finish rendering the equations
                    f"--print-to-pdf={pdf}", "file:///" + tmp.replace("\\", "/")],
                   check=True, capture_output=True)
    print("wrote", pdf)
