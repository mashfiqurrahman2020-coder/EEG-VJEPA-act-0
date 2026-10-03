"""analysis.md + numbers.json/tables.json -> Act0_Results_Analysis.pdf. Run make_results.py first.
{{key}} -> numbers.json, {{t:name}} -> tables.json; any unknown key raises."""
import json, os, re
import markdown
from weasyprint import HTML

D = os.path.dirname(os.path.abspath(__file__))
N = json.load(open(f"{D}/numbers.json"))
T = json.load(open(f"{D}/tables.json"))


def sub(m):
    k = m.group(1)
    if k.startswith("t:"):
        return "\n\n" + T[k[2:]] + "\n\n"   # KeyError = missing table
    return str(N[k])                          # KeyError = missing number


md = re.sub(r"\{\{([^}]+)\}\}", sub, open(f"{D}/analysis.md", encoding="utf-8").read())
assert "{{" not in md
body = markdown.markdown(md, extensions=["tables"])
# figure captions: markdown alt text -> visible caption under the image
body = re.sub(r'<p><img alt="([^"]*)" src="([^"]+)" ?/?></p>',
              r'<figure><img src="\2"><figcaption>\1</figcaption></figure>', body)

# short bold-only paragraphs ("Setup.", "Result.") are headings: keep them with what follows
body = re.sub(r"<p>(<strong>[^<]{1,60}</strong>)</p>", r'<p class="lead">\1</p>', body)

CSS = """
@page { size: A4; margin: 16mm; @bottom-center { content: counter(page); font-size: 9pt; color: #888; } }
body { font-family: Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; }
h1 { color: #333; page-break-before: always; border-bottom: 2px solid #ccc; }
h2 { color: #333; margin-top: 1.6em; page-break-after: avoid; }
h3 { color: #333; page-break-after: avoid; }
figure { margin: 0.6em 0; page-break-inside: avoid; }
figure img { max-width: 100%; max-height: 110mm; height: auto; display: block; margin: 0 auto; }
img[src$="stroke_perm.png"] { max-width: 55%; }
img[src$="stroke_clean_cv.png"] { max-width: 80%; }
img[src$="tuab_maps.png"] { max-width: 82%; }
figcaption { font-size: 9pt; color: #555; text-align: center; margin-top: 0.2em; }
table { border-collapse: collapse; margin: 0.6em 0; font-size: 9pt; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 3px 6px; vertical-align: top; }
th { background: #eee; }
code { font-size: 8.5pt; background: #f3f3f3; }
blockquote { border-left: 4px solid #2a7; background: #eaf7ef; margin: 0.8em 0; padding: 6px 10px; page-break-inside: avoid; }
blockquote p { margin: 0; }
.warn { border-left: 4px solid #c33; background: #fbeaea; padding: 6px 10px; margin: 0.8em 0; }
.lead { page-break-after: avoid; }
.cover { text-align: center; padding-top: 55mm; }
.cover .title { font-size: 24pt; font-weight: bold; }
.cover .sub { font-size: 13pt; color: #555; margin-bottom: 30mm; }
.cover .warn, .cover .note { text-align: left; }
.cover .note { font-size: 9.5pt; color: #555; }
"""
out = f"{D}/Act0_Results_Analysis.pdf"
HTML(string=f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>",
     base_url=D).write_pdf(out)
print(out)
