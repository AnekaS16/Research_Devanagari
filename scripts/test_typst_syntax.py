import typst

# Test 1: bibliography
with open('test_bib.typ', 'w', encoding='utf-8') as f:
    f.write(r"""
#set page(paper: "a4")
= Test
Some text with @devlin2019bert.
#bibliography("research_paper/references.bib", title: "References")
""")
try:
    typst.compile('test_bib.typ', output='test_bib.pdf')
    print('bib OK')
except typst.TypstError as e:
    print(f'bib ERROR: {e}')

# Test 2: hanging-indent (added in Typst 0.11)
with open('test_par.typ', 'w', encoding='utf-8') as f:
    f.write(r"""
#set page(paper: "a4")
#set par(justify: true, leading: 0.65em, first-line-indent: 1.2em, hanging-indent: 1.2em)
= Test
Some text here.
""")
try:
    typst.compile('test_par.typ', output='test_par.pdf')
    print('hanging-indent OK')
except typst.TypstError as e:
    print(f'hanging-indent ERROR: {e}')

# Test 3: table.header
with open('test_table.typ', 'w', encoding='utf-8') as f:
    f.write(r"""
#set page(paper: "a4")
#table(
  columns: (auto, auto),
  inset: 5pt,
  table.header([*A*], [*B*]),
  [row1a], [row1b],
)
""")
try:
    typst.compile('test_table.typ', output='test_table.pdf')
    print('table.header OK')
except typst.TypstError as e:
    print(f'table.header ERROR: {e}')

# Test 4: math.inline
with open('test_mathinline.typ', 'w', encoding='utf-8') as f:
    f.write(r"""
#set page(paper: "a4")
Test #math.inline[$N=200$] here.
""")
try:
    typst.compile('test_mathinline.typ', output='test_mathinline.pdf')
    print('math.inline OK')
except typst.TypstError as e:
    print(f'math.inline ERROR: {e}')
