"""
Check for cite() function usage vs @ syntax and other content-specific issues.
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

# Check for #cite() usage (vs @citekey)
cite_uses = re.findall(r'#cite\(<[^>]+>', content)
print(f"#cite() calls found: {len(cite_uses)}")

# Replace all #cite(<key>) with @key style
def cite_to_at(c):
    # #cite(<key1>, <key2>) -> @key1 @key2
    def replace_cite(m):
        keys = re.findall(r'<([^>]+)>', m.group(0))
        return ' '.join(f'@{k}' for k in keys)
    return re.sub(r'#cite\([^)]+\)', replace_cite, c)

c_at_cite = cite_to_at(content)
# Also replace bib with one that exists
c_at_cite_nobib = c_at_cite.replace(
    '#bibliography("../research_paper/references.bib", title: none, style: "apa")',
    '#bibliography("research_paper/references.bib", title: none)'
)
test("@-cite style + bib path fixed", c_at_cite_nobib)

# Test with no citation at all
c_no_cite = re.sub(r'#cite\([^)]+\)', '', content)
c_no_cite = c_no_cite.replace(
    '#bibliography("../research_paper/references.bib", title: none, style: "apa")', '')
test("no citations at all", c_no_cite)

# Check for #cite with multiple args
multi_cite = re.findall(r'#cite\([^)]+,[^)]+\)', content)
print(f"\nMulti-arg #cite() calls: {len(multi_cite)}")
for m in multi_cite[:5]:
    print(f"  {m}")
