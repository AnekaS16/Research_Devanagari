"""
Progressive inclusion test to isolate the issue.
"""
import typst

def test(name, c):
    with open('_test.typ', 'w', encoding='utf-8') as f:
        f.write(c)
    try:
        typst.compile('_test.typ', output='_test.pdf')
        print(f"[OK] {name}")
        return True
    except typst.TypstError as e:
        print(f"[ERR] {name}: {e}")
        return False

# Minimal version
test("minimal", '#set page(paper: "a4")\n= Hello\nWorld')

# Page with header/footer
test("header+footer", '''
#set page(
  paper: "a4",
  header: align(center, text(size: 8pt)[My header]),
  footer: context align(center, text(size: 8pt)[
    #counter(page).display()
  ]),
)
= Hello
World
''')

# With math equation numbering
test("math numbered", '''
#set page(paper: "a4")
#set math.equation(numbering: "(1)")
= Hello
$ sum_(i=1)^N x_i $ <eq1>
See @eq1.
''')

# heading.where show rule
test("show heading.where", '''
#set page(paper: "a4")
#show heading.where(level: 1): it => {
  v(1.2em)
  text(size: 11.5pt, weight: "bold")[#it]
  v(0.4em)
}
= Hello
World
''')

# Table with fill lambda
test("table fill lambda", '''
#set page(paper: "a4")
#table(
  columns: (auto, auto),
  inset: 5pt,
  stroke: 0.5pt + luma(180),
  fill: (col, row) => if row == 0 { rgb("#1a365d") } else if calc.odd(row) { rgb("#f8fafc") } else { white },
  table.header(
    text(fill: white, weight: "bold")[Col A],
    text(fill: white, weight: "bold")[Col B],
  ),
  [a], [b],
  [c], [d],
)
''')

# Table with fill lambda (underscore parameter)
test("table fill lambda underscore", '''
#set page(paper: "a4")
#table(
  columns: (auto, auto),
  inset: 5pt,
  fill: (_, row) => if row == 0 { rgb("#1a365d") } else if calc.odd(row) { rgb("#f8fafc") } else { white },
  table.header(
    [*Col A*],
    [*Col B*],
  ),
  [a], [b],
  [c], [d],
)
''')

# columns block
test("columns block", '''
#set page(paper: "a4")
#columns(2, gutter: 1em)[
= Section One
Content in column one.
= Section Two
Content in column two.
]
''')

# cite
test("cite", '''
#set page(paper: "a4")
Some text @devlin2019bert.
#bibliography("research_paper/references.bib")
''')
