"""
Render first 4 pages of research_paper_refined.pdf to PNG for visual inspection.
"""
import fitz  # PyMuPDF
import os

PDF = r"D:\hindi-lexical-sensitivity\research_paper\refined\research_paper_refined.pdf"
OUT_DIR = r"D:\hindi-lexical-sensitivity\research_paper\refined\preview"
os.makedirs(OUT_DIR, exist_ok=True)

doc = fitz.open(PDF)
print(f"Total pages: {len(doc)}")

for i in range(min(4, len(doc))):
    page = doc[i]
    mat = fitz.Matrix(2.0, 2.0)  # 2x zoom = ~144 DPI
    pix = page.get_pixmap(matrix=mat)
    out_path = os.path.join(OUT_DIR, f"page_{i+1:02d}.png")
    pix.save(out_path)
    print(f"  Saved: {out_path} ({pix.width}x{pix.height})")

doc.close()
print("Done.")
