"""Render a markdown document to a self-contained PDF with figures embedded.

Markdown -> styled HTML (images inlined as base64) -> headless Chrome --print-to-pdf.

Chrome rather than pandoc/WeasyPrint deliberately: WeasyPrint needs GTK/cairo/pango native
libraries that are painful to install on Windows, pandoc needs a LaTeX distribution, and
both are heavier dependencies than a repo that already assumes a browser is present for the
dashboard. Inlining the images as data URIs means the intermediate HTML has no external
file references, so Chrome renders it identically regardless of working directory.

    python scripts/render_report.py            # -> reports/technical_report.pdf
    python scripts/render_report.py --slides \\
        --md reports/viva_deck.md --pdf reports/viva_deck.pdf

`--slides` switches to landscape with one slide per `---` rule and larger type.
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

# Landscape, one slide per horizontal rule, type sized to be legible from the back of a
# room rather than to fit the most words on a page.
SLIDES_CSS = """
@page { size: A4 landscape; margin: 14mm 16mm; }
body {
  font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  font-size: 14pt; line-height: 1.45; color: #16181d;
}
.slide { page-break-after: always; min-height: 172mm; }
.slide:last-child { page-break-after: auto; }
hr { display: none; }
h1 { font-size: 30pt; margin: 0 0 6px; color: #10131a; line-height: 1.15; }
h2 { font-size: 21pt; margin: 0 0 10px; color: #10131a; border-bottom: 2px solid #d8dbe2;
     padding-bottom: 6px; line-height: 1.25; }
h3 { font-size: 15pt; margin: 14px 0 6px; }
p { margin: 8px 0; }
ul, ol { margin: 8px 0; padding-left: 22px; }
li { margin: 5px 0; }
strong { color: #000; }
code { background: #eef0f4; padding: 1px 5px; border-radius: 3px;
       font-family: "Cascadia Mono", Consolas, monospace; font-size: 12pt; }
table { border-collapse: collapse; margin: 12px 0; font-size: 12.5pt; width: 100%; }
th, td { border: 1px solid #c2c7d0; padding: 6px 10px; text-align: left; }
th { background: #eef0f4; }
blockquote { border-left: 4px solid #c9ced8; margin: 10px 0 0; padding: 4px 0 4px 14px;
             color: #555; font-size: 11.5pt; }
img { max-width: 100%; max-height: 105mm; height: auto; display: block; margin: 8px auto; }
.figure { text-align: center; margin: 8px 0; }
.figure .caption { font-size: 10pt; color: #555; font-style: italic; margin-top: 3px; }
"""


def split_slides(html: str) -> str:
    """Wrap the content between <hr /> rules in page-breaking .slide divs."""
    parts = re.split(r"<hr\s*/?>", html)
    return "".join(f'<div class="slide">{p}</div>' for p in parts if p.strip())


def normalise_pdf_dates(pdf: Path) -> None:
    """Pin the embedded /CreationDate and /ModDate to a fixed epoch.

    Chrome stamps the current time into every PDF, so re-rendering unchanged markdown
    produces different bytes and shows up as a modified 680KB binary in git -- the same
    class of meaningless churn .gitattributes exists to prevent for text. Verified that
    timestamps are the *only* non-determinism: two renders are byte-identical once these
    fields match.

    The replacement is the same byte length as what it replaces, deliberately: a PDF's
    xref table stores absolute byte offsets, so changing the length would corrupt the file.
    """
    raw = pdf.read_bytes()
    fixed = re.sub(
        rb"(/(?:CreationDate|ModDate)\s*\(D:)\d{14}",
        rb"\g<1>19700101000000",
        raw,
    )
    if fixed != raw:
        assert len(fixed) == len(raw), "date normalisation changed byte length"
        pdf.write_bytes(fixed)


def _display(path: Path) -> str:
    """Repo-relative path for logging, falling back to absolute if outside the repo."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


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
    ap.add_argument(
        "--slides",
        action="store_true",
        help="landscape slide deck: one slide per '---' rule, larger type",
    )
    args = ap.parse_args()

    # Resolve so paths given relative to the caller's cwd still work, and so the
    # relative_to(ROOT) display below can't raise on an outside-the-repo path.
    args.md = args.md.resolve()
    args.pdf = args.pdf.resolve()

    if not args.md.exists():
        raise SystemExit(f"not found: {args.md}")

    print(f"rendering {_display(args.md)}")
    body = markdown.markdown(
        args.md.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    )
    body = inline_images(body, args.md.parent)
    if args.slides:
        body = split_slides(body)
        print(f"  {body.count('class="slide"')} slide(s)")
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{args.md.stem}</title>"
        f"<style>{SLIDES_CSS if args.slides else CSS}</style></head>"
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

    normalise_pdf_dates(args.pdf)

    print(f"wrote {_display(args.pdf)} ({args.pdf.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
