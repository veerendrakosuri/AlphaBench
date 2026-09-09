"""Render reports/technical_report.md to a self-contained PDF with figures embedded.

Markdown -> styled HTML (images inlined as base64) -> headless Chrome --print-to-pdf.

Chrome rather than pandoc/WeasyPrint deliberately: WeasyPrint needs GTK/cairo/pango native
libraries that are painful to install on Windows, pandoc needs a LaTeX distribution, and
both are heavier dependencies than a repo that already assumes a browser is present for the
dashboard. Inlining the images as data URIs means the intermediate HTML has no external
file references, so Chrome renders it identically regardless of working directory.

    python scripts/render_report.py            # -> reports/technical_report.pdf
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MD = ROOT / "reports" / "technical_report.md"
DEFAULT_PDF = ROOT / "reports" / "technical_report.pdf"

CHROME_CANDIDATES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path("/usr/bin/google-chrome"),
    Path("/usr/bin/chromium"),
    Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
]

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body {
  font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  font-size: 10.5pt; line-height: 1.5; color: #1a1a1a; max-width: 100%;
}
h1 { font-size: 20pt; border-bottom: 2px solid #333; padding-bottom: 6px; margin-top: 0; }
h2 { font-size: 14pt; margin-top: 22px; border-bottom: 1px solid #ccc; padding-bottom: 3px;
     page-break-after: avoid; }
h3 { font-size: 11.5pt; margin-top: 16px; page-break-after: avoid; }
p, li { orphans: 3; widows: 3; }
code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px;
       font-family: "Cascadia Mono", Consolas, monospace; font-size: 9pt; }
pre { background: #f4f4f4; padding: 9px; border-radius: 4px; overflow-x: auto;
      page-break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; margin: 12px 0; font-size: 9.5pt;
        page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 5px 9px; text-align: left; }
th { background: #eee; }
blockquote { border-left: 3px solid #bbb; margin-left: 0; padding-left: 12px; color: #444; }
img { max-width: 100%; height: auto; display: block; margin: 14px auto; }
/* Markdown image alt text becomes the figure caption, rendered under the image. */
.figure { page-break-inside: avoid; margin: 16px 0; text-align: center; }
.figure .caption { font-size: 8.5pt; color: #555; font-style: italic;
                   margin-top: 4px; text-align: left; }
"""


def find_browser() -> Path:
    for c in CHROME_CANDIDATES:
        if c.exists():
            return c
    for name in ("google-chrome", "chromium", "chrome", "msedge"):
        found = shutil.which(name)
        if found:
            return Path(found)
    raise SystemExit(
        "No Chrome/Chromium/Edge found for PDF printing. Install one, or render the "
        "markdown yourself -- reports/technical_report.md is the source of truth."
    )


def inline_images(html: str, base_dir: Path) -> str:
    """Replace <img src="..."> with base64 data URIs and attach the alt text as a caption."""

    def repl(match: re.Match[str]) -> str:
        src = match.group("src")
        alt = match.group("alt") or ""
        path = (base_dir / src).resolve()
        if not path.exists():
            raise SystemExit(f"figure referenced but not found: {src} (looked in {path})")
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        b64 = base64.b64encode(path.read_bytes()).decode()
        caption = f'<div class="caption">{alt}</div>' if alt else ""
        return (
            f'<div class="figure"><img src="data:{mime};base64,{b64}" alt="{alt}" />{caption}</div>'
        )

    pattern = re.compile(r'<img alt="(?P<alt>[^"]*)" src="(?P<src>[^"]+)"\s*/?>')
    html, n = pattern.subn(repl, html)
    print(f"  embedded {n} figure(s)")
    return html


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--md", type=Path, default=DEFAULT_MD)
    ap.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    args = ap.parse_args()

    if not args.md.exists():
        raise SystemExit(f"not found: {args.md}")

    print(f"rendering {args.md.relative_to(ROOT)}")
    body = markdown.markdown(
        args.md.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    )
    body = inline_images(body, args.md.parent)
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{args.md.stem}</title><style>{CSS}</style></head>"
        f"<body>{body}</body></html>"
    )

    browser = find_browser()
    print(f"  printing via {browser.name}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_html = Path(tmp) / "report.html"
        tmp_html.write_text(html, encoding="utf-8")
        args.pdf.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                str(browser),
                "--headless",
                "--disable-gpu",
                "--no-sandbox",
                "--no-pdf-header-footer",
                f"--print-to-pdf={args.pdf}",
                tmp_html.as_uri(),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
    if not args.pdf.exists():
        print(result.stdout, result.stderr, file=sys.stderr)
        raise SystemExit("Chrome did not produce a PDF")

    print(f"wrote {args.pdf.relative_to(ROOT)} ({args.pdf.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
