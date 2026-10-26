"""Render FAILURE_CASES.md -> FAILURE_CASES.pdf with weasyprint (contact sheets on landscape pages)."""
import re, markdown, weasyprint
D='/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING/results/failure_cases'
md=open(f'{D}/FAILURE_CASES.md').read()
html=markdown.markdown(md,extensions=['tables'])
html=re.sub(r'<p><img alt="([^"]*)" src="([^"]*)" ?/?></p>',
            r'<figure class="sheet"><img src="\2"/><figcaption>\1 <code>\2</code></figcaption></figure>',html)
css='''
@page{size:A4;margin:14mm 13mm;@bottom-center{content:counter(page);font-size:8pt;color:#666}}
@page land{size:A4 landscape;margin:8mm}
body{font-family:"DejaVu Sans",sans-serif;font-size:8.6pt;line-height:1.38;color:#111}
h1{font-size:15pt;margin:0 0 6pt} h2{font-size:12pt;border-bottom:1px solid #999;margin-top:14pt}
h3{font-size:10pt;margin-top:10pt}
table{border-collapse:collapse;width:100%;margin:5pt 0;font-size:7.4pt;page-break-inside:avoid}
th,td{border:1px solid #bbb;padding:2pt 3pt;vertical-align:top} th{background:#eee}
code{font-size:7.4pt;background:#f3f3f3}
figure.sheet{page:land;page-break-before:always;page-break-after:always;margin:0;text-align:center}
figure.sheet img{max-width:100%;max-height:178mm}
figcaption{font-size:9pt;margin-top:3pt}
li{margin:1pt 0}
'''
doc=f'<html><head><meta charset="utf-8"><style>{css}</style></head><body>{html}</body></html>'
weasyprint.HTML(string=doc,base_url=D+'/').write_pdf(f'{D}/FAILURE_CASES.pdf')
print('ok')
