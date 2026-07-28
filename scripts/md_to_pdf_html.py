#!/usr/bin/env python3
"""Convert the paper markdown to a self-contained HTML for browser->PDF export.

- embeds images as base64 (HTML is portable, no external file deps)
- MathJax (CDN) renders $$...$$ math
- print-friendly CSS (A4, margins)
Usage: python scripts/md_to_pdf_html.py paper/draft_3pillars_cn.md
Then open the .html in a browser, Ctrl+P, save as PDF.
"""
import base64
import re
import sys
from pathlib import Path

import markdown

HTML_TMPL = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>{title}</title>
<script>
window.MathJax = {{
  tex: {{ inlineMath: [['$','$'],['\\\\(','\\\\)']],
          displayMath: [['$$','$$'],['\\\\[','\\\\]']] }},
  svg: {{ fontCache: 'global' }}
}};
</script>
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<style>
@page {{ size: A4; margin: 18mm; }}
body {{ font-family: "Noto Sans CJK SC","Noto Sans CJK JP","Source Han Sans SC",
       "Microsoft YaHei", sans-serif; max-width: 760px; margin: 0 auto;
       line-height: 1.65; color: #1a1a1a; font-size: 11.5pt; }}
h1 {{ font-size: 20pt; border-bottom: 2px solid #333; padding-bottom: 6px; }}
h2 {{ font-size: 15pt; margin-top: 1.4em; border-bottom: 1px solid #ccc; padding-bottom: 3px; }}
h3 {{ font-size: 13pt; margin-top: 1.2em; }}
table {{ border-collapse: collapse; width: 100%; margin: 0.8em 0; font-size: 10.5pt; }}
th, td {{ border: 1px solid #888; padding: 5px 8px; text-align: center; }}
th {{ background: #f0f0f0; }}
img {{ max-width: 100%; display: block; margin: 0.6em auto; }}
code, pre {{ font-family: "JetBrains Mono","Consolas",monospace; font-size: 10pt; }}
pre {{ background: #f6f6f6; padding: 10px; border-radius: 4px; overflow-x: auto;
       white-space: pre-wrap; }}
blockquote {{ border-left: 3px solid #aaa; margin: 0.6em 0; padding: 0.2em 0.9em;
             color: #555; background: #fafafa; }}
@media print {{ body {{ max-width: none; }} }}
</style></head><body>
{body}
</body></html>
"""


def embed_images(md_text, md_dir):
    def repl(m):
        alt, path = m.group(1), m.group(2)
        p = Path(path)
        if not p.is_absolute():
            p = md_dir / p
        if not p.exists():
            return m.group(0)
        mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
        b64 = base64.b64encode(p.read_bytes()).decode()
        return f'<img alt="{alt}" src="data:{mime};base64,{b64}"/>'
    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", repl, md_text)


def main():
    if len(sys.argv) < 2:
        print("usage: md_to_pdf_html.py <file.md>"); sys.exit(1)
    md_path = Path(sys.argv[1])
    md_text = md_path.read_text(encoding="utf-8")
    md_text = embed_images(md_text, md_path.parent)
    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "md_in_html"],
    )
    title = md_path.stem
    html = HTML_TMPL.format(title=title, body=body)
    out = md_path.with_suffix(".html")
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size//1024} KB)")
    print("open in browser -> Ctrl+P -> save as PDF")


if __name__ == "__main__":
    main()
