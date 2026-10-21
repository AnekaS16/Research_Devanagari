"""
Binary-search the .typ file to find the first line causing the compile error.
"""
import typst
import os

TYP_FILE = r"D:\hindi-lexical-sensitivity\research_paper\refined\research_paper_refined.typ"
PREAMBLE = r"""#set page(paper: "a4")
"""

with open(TYP_FILE, encoding='utf-8') as f:
    lines = f.readlines()

total = len(lines)
print(f"Total lines: {total}")

def try_compile(content):
    with open('test_chunk.typ', 'w', encoding='utf-8') as f:
        f.write(content)
    try:
        typst.compile('test_chunk.typ', output='test_chunk.pdf')
        return True, None
    except typst.TypstError as e:
        return False, str(e)

# Binary search
lo, hi = 0, total
while lo < hi:
    mid = (lo + hi) // 2
    content = ''.join(lines[:mid])
    ok, err = try_compile(content)
    if ok:
        lo = mid + 1
    else:
        hi = mid
    print(f"  lo={lo} hi={hi} mid={mid} ok={ok} err={err}")

print(f"\nFirst failing line: {lo}")
if lo < total:
    print(f"Line {lo}: {lines[lo-1].rstrip()}")
