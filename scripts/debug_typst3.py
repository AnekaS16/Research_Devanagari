"""
Pinpoint the 'unexpected argument' error by systematic exclusion.
"""
import typst, re

TYP_FILE = r"D:\hindi-lexical-sensitivity\research_paper\refined\research_paper_refined.typ"
with open(TYP_FILE, encoding='utf-8') as f:
    content = f.read()

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

# Test 1: remove context (footer uses it)
c1 = content.replace(
    "footer: context align(center, text(size: 8pt, fill: luma(120))[\n    #counter(page).display()\n  ]),",
    "footer: align(center, text(size: 8pt, fill: luma(120))[page]),"
)
test("no context footer", c1)

# Test 2: remove all images
c2 = re.sub(r'#image\([^)]+\)', '', content)
test("no images", c2)

# Test 3: remove eq.not
c3 = content.replace('eq.not', 'eq.not')
test("eq.not unchanged", c3)

# Test 4: remove bb() usage
c4 = content.replace('bb(1)(Delta S_"top",i > Delta S_"bottom",i)', '1(Delta S_"top",i > Delta S_"bottom",i)')
test("no bb", c4)

# Test 5: remove context footer + no images
c5 = re.sub(r'#image\([^)]+\)', '', c1)
test("no context + no images", c5)

# Test 6: remove colbreak
c6 = content.replace('#colbreak()', '')
test("no colbreak", c6)

# Test 7: remove heading.where show rules
c7 = re.sub(r'#show heading\.where\([^}]+\}\n\}', '', content, flags=re.DOTALL)
test("no show heading.where", c7)

# Test 8: check bibliography style arg
c8 = content.replace('style: "apa"', '')
test("bib without style", c8)

# Test 9: remove the #set par hanging-indent in References
c9 = content.replace('#set par(first-line-indent: 0em, hanging-indent: 1.2em)', '#set par(first-line-indent: 0em)')
test("no hanging-indent in ref", c9)

# Test 10: remove context footer + no images + no bib
c10 = re.sub(r'#image\([^)]+\)', '', c1)
c10 = c10.replace('#bibliography("../research_paper/references.bib", title: none, style: "apa")', '= References\n\n_Bibliography placeholder._')
test("no context + no images + no bib", c10)
