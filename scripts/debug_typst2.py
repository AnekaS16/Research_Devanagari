"""
Find the 'unexpected argument' error in the .typ file by testing progressively more content.
"""
import typst

TYP_FILE = r"D:\hindi-lexical-sensitivity\research_paper\refined\research_paper_refined.typ"

with open(TYP_FILE, encoding='utf-8') as f:
    content = f.read()

# Test each potential culprit
suspects = [
    # table fill with callable
    ('fill arg', content.find('fill: (_, row) =>')),
    # bibliography style
    ('bib style', content.find('style: "apa"')),
    # heading.where
    ('show heading', content.find('#show heading.where')),
    # context
    ('context', content.find('context align')),
    # math.inline
    ('math.inline', content.find('#math.inline')),
    # colbreak
    ('colbreak', content.find('#colbreak()')),
    # bb (blackboard bold)
    ('bb(1)', content.find('bb(1)')),
    # eq.not
    ('eq.not', content.find('eq.not')),
    # table.header
    ('table.header', content.find('table.header(')),
    # hanging-indent in set par
    ('hanging-indent', content.find('hanging-indent')),
]

print("Suspected problematic lines:")
lines = content.split('\n')
for name, pos in suspects:
    if pos >= 0:
        line_no = content[:pos].count('\n') + 1
        print(f"  [{name}] line {line_no}: {lines[line_no-1].strip()[:80]}")
    else:
        print(f"  [{name}] NOT FOUND")

# Test: replace fill lambda with solid fill
print("\n--- Testing table without lambda fill ---")
import re
test_content = re.sub(
    r'fill: \([^)]+\) => [^\n]+,\n',
    'fill: white,\n',
    content
)
with open('test_nolambda.typ', 'w', encoding='utf-8') as f:
    f.write(test_content)
try:
    typst.compile('test_nolambda.typ', output='test_nolambda.pdf')
    print("Without lambda fill: OK")
except typst.TypstError as e:
    print(f"Without lambda fill: ERROR {e}")

# Test: remove bibliography
print("--- Testing without bibliography ---")
test_content2 = content.replace(
    '#bibliography("../research_paper/references.bib", title: none, style: "apa")',
    ''
)
with open('test_nobib.typ', 'w', encoding='utf-8') as f:
    f.write(test_content2)
try:
    typst.compile('test_nobib.typ', output='test_nobib.pdf')
    print("Without bib: OK")
except typst.TypstError as e:
    print(f"Without bib: ERROR {e}")
