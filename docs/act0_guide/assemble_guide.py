import ast
import os
import re
import markdown
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer
from weasyprint import HTML

# Directory paths
base_dir = "/home/mashfiq/eeg_vjepa/docs/act0_guide"
new_sections_dir = os.path.join(base_dir, "new_sections")
output_pdf_path = os.path.join(base_dir, "Act0_Teaching_Guide.pdf")

CODE_DIR = "/home/mashfiq/eeg_vjepa/code"
MAX_LINES = 40   # keep every excerpt short enough to read on one page


def code_ranges(path, spec):
    """spec ':Name' / ':Class.method' -> that definition's lines (via ast); '57-71,90' -> those lines."""
    if spec.startswith(":"):
        node = ast.parse(open(f"{CODE_DIR}/{path}").read())
        for part in spec[1:].split("."):
            node = next((n for n in ast.iter_child_nodes(node) if getattr(n, "name", None) == part), None)
            if node is None:
                raise SystemExit(f"code snippet: {part} not found in {path}")
        return [(min([node.lineno] + [d.lineno for d in node.decorator_list]), node.end_lineno)]
    return [tuple(map(int, r.split("-"))) if "-" in r else (int(r), int(r)) for r in spec.split(",")]


def code_block(path, spec):
    """Verbatim excerpt of code/<path> (copied at build time, never retyped) + a file:line caption."""
    lines = open(f"{CODE_DIR}/{path}").read().splitlines()
    rng = code_ranges(path, spec)
    chunks = [lines[a - 1:b] for a, b in rng]
    assert sum(map(len, chunks)) <= MAX_LINES, (path, spec)
    ind = min(len(l) - len(l.lstrip()) for c in chunks for l in c if l.strip())
    body = "\n    ...\n".join("\n".join(l[ind:] for l in c) for c in chunks)   # '...' marks skipped lines
    where = ", ".join(f"{a}" if a == b else f"{a}–{b}" for a, b in rng)
    cap = f"code/{path}, line{'s' if len(rng) > 1 or rng[0][0] != rng[0][1] else ''} {where}"
    return (f'<div class="snip"><div class="cap">{cap}</div>'
            + highlight(body, PythonLexer(), HtmlFormatter(noclasses=True, style="friendly")) + "</div>")


# Files to process in order
files = ["nmt.md", "tuab.md", "stroke.md", "vjepa.md", "feic.md", "eval_architectures.md"]

assembled_md = """<div class="cover">
<p class="title">Act-0 Data Pipelines and Architectures</p>
<p class="sub">A teaching guide for the team: NMT, TUAB and stroke data, EEG-VJEPA, FEI+C and the evaluation heads</p>
<p>EEE 402 (G1) — Artificial Intelligence and Machine Learning Laboratory<br>Group 01 · Bangladesh University of Engineering and Technology (BUET)</p>
<div class="warn">Internal team document. The TUAB numbers come from an unlicensed Kaggle copy and must not be shared outside the group.</div>
<p><b>Contents:</b> 1. NMT · 2. TUAB · 3. Stroke · 4. V-JEPA · 5. FEI+C · 6. Evaluation heads</p>
<p class="note">The grey code boxes are copied word for word from the repository when this PDF is built. Each box names its file (under <code>code/</code>) and line numbers, so you can open the same lines in the editor. <code>...</code> on its own line marks skipped lines. Files under <code>src/</code> and <code>app/</code> are the EEG-VJEPA authors' code; the rest is ours.</p>
</div>

"""

for filename in files:
    filepath = os.path.join(new_sections_dir, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        assembled_md += f.read() + "\n\n"

# {{code:PATH:SPEC}} on its own line -> placeholder paragraph, swapped for the highlighted block after markdown
snips = []
def _stash(m):
    snips.append(code_block(m.group(1), m.group(2)))
    return f"@@CODE{len(snips) - 1}@@"
assembled_md = re.sub(r"\{\{code:([^:}]+):([^}]+)\}\}", _stash, assembled_md)

# Convert Markdown to HTML
html_content = markdown.markdown(assembled_md, extensions=["tables"])
for i, b in enumerate(snips):
    # an intro paragraph ending in ':' right before a box goes inside it, so a page break never splits them
    html_content = re.sub(rf"(<p>(?:(?!</?p>).)*:</p>)\s*<p>@@CODE{i}@@</p>",
                          lambda m: b.replace('<div class="snip">', '<div class="snip">' + m.group(1), 1),
                          html_content, flags=re.S)
    html_content = html_content.replace(f"<p>@@CODE{i}@@</p>", b)
assert "@@CODE" not in html_content, "a code directive was not on its own line"

# Minimal CSS to ensure images fit the page and there are page breaks if needed
css_style = """
<style>
    @page { size: A4; margin: 16mm; @bottom-center { content: counter(page); font-size: 9pt; color: #888; } }
    body { font-family: Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; }
    h1 { color: #333; page-break-before: always; border-bottom: 2px solid #ccc; }
    h2, h3 { color: #333; page-break-after: avoid; }
    img { max-width: 100%; max-height: 120mm; height: auto; display: block; margin: 0.6em auto; page-break-inside: avoid; }
    table { border-collapse: collapse; margin: 0.6em 0; font-size: 9.5pt; page-break-inside: avoid; }
    th, td { border: 1px solid #bbb; padding: 3px 6px; vertical-align: top; }
    th { background: #eee; }
    code { font-size: 9pt; background: #f3f3f3; }
    .warn { border-left: 4px solid #c33; background: #fbeaea; padding: 6px 10px; margin: 0.8em 0; }
    .cover { text-align: center; padding-top: 60mm; }
    .cover .title { font-size: 24pt; font-weight: bold; }
    .cover .sub { font-size: 13pt; color: #555; margin-bottom: 30mm; }
    .cover .warn { text-align: left; }
    .cover .note { text-align: left; font-size: 9.5pt; color: #444; }
    .snip { page-break-inside: avoid; margin: 0.4em 0 0.9em; }
    .snip .cap { font-family: monospace; font-size: 7.5pt; color: #666; margin-bottom: 1px; }
    .snip .highlight { border-left: 3px solid #8a9bb0; }
    .snip pre { font-size: 7.6pt; line-height: 1.3; padding: 5px 8px; margin: 0; white-space: pre-wrap; }
    .snip pre code, .snip pre span { background: none; font-size: inherit; }
</style>
"""
full_html = f"<html><head><meta charset='utf-8'>{css_style}</head><body>{html_content}</body></html>"

# Use weasyprint to convert to PDF
HTML(string=full_html, base_url=base_dir).write_pdf(output_pdf_path)

print(f"Successfully created {output_pdf_path}")
