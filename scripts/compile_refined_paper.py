"""
Compile research_paper_refined.typ into research_paper_refined.pdf using the Typst Python API.
Outputs: research_paper/refined/research_paper_refined.pdf
"""
import typst
import sys
import os

TYP_FILE = r"D:\hindi-lexical-sensitivity\research_paper\refined\research_paper_refined.typ"
PDF_OUT  = r"D:\hindi-lexical-sensitivity\research_paper\refined\research_paper_refined.pdf"

os.chdir(r"D:\hindi-lexical-sensitivity")

print(f"[compile] Compiling {TYP_FILE} ...")
try:
    typst.compile(
        TYP_FILE,
        output=PDF_OUT,
    )
    size = os.path.getsize(PDF_OUT)
    print(f"[compile] OK  PDF written to: {PDF_OUT}  ({size / 1024:.1f} KB)")
except typst.TypstError as e:
    print(f"[compile] ERROR: {e}")
    sys.exit(1)
