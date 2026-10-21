"""Package the independently checked reference study for the dashboard's read-only link."""
import csv
import hashlib
import html
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'docs/artifacts/capstone'
OUTPUT = ROOT / 'frontend/public/capstone/study-v1'
TRUSTED = {
    'nih-phase2-study-v1-verification.json': 'b875362354613023d4adad53be26f48b8fd37c6ca1a7b6d158d18e8871e97ff7',
    'nih-phase2-study-v1-report.md': '4b48870e4353767fe78ebc1593f52f085791329579b5f5c0a6a1c674b522999f',
    'tables/nih-phase2-v1/table-registry.json': '3e5126627abf50a7816976c5a48f991f068a8b1e9145877f4564df86056b35d6',
    'figures/nih-phase2-figure-values.json': '34c9d924329a77090e9c2b08837cb37aa2bce4d447dd77534ffe952678d4e00a',
}
# Immutable reviewed delivery roots from 4e80033. Corrections require a new version.
for name, digest in TRUSTED.items():
    if hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() != digest:
        raise ValueError(f'Reference source identity changed: {name}')
def require(condition, message):
    if not condition:
        raise ValueError(message)


verification = json.loads((SOURCE / 'nih-phase2-study-v1-verification.json').read_text())
tables = json.loads((SOURCE / 'tables/nih-phase2-v1/table-registry.json').read_text())
plots = json.loads((SOURCE / 'figures/nih-phase2-figure-values.json').read_text())
for registry in (tables, plots):
    require(registry['source_commit'] == verification['source_commit'], 'Registry source mismatch')
    require(registry['independent_audit_sha256'] == verification['independent_audit_sha256'], 'Registry audit mismatch')
table = SOURCE / 'tables/nih-phase2-v1/all_64_recipe_groups.csv'
require(hashlib.sha256(table.read_bytes()).hexdigest() == tables['files'][table.name]['sha256'], 'Table content changed')
require(verification['status'] == 'independent_source_bound_and_review_corrections_verified', 'Unverified study')
require(verification['qualified'] == 360 and verification['verified_rounds'] == 7200, 'Incomplete study')
require(verification['failed'] == verification['timeout'] == verification['not_run'] == 0, 'Study has unresolved outcomes')
with (SOURCE / 'tables/nih-phase2-v1/all_64_recipe_groups.csv').open() as stream:
    groups = list(csv.DictReader(stream))
require(len(groups) == 64, 'Incomplete group table')
sources = [SOURCE / 'nih-phase2-study-v1-verification.json',
           SOURCE / 'nih-phase2-study-v1-report.md',
           SOURCE / 'tables/nih-phase2-v1/all_64_recipe_groups.csv']
figures = [('nih-phase2-primary-components.png', 'Absolute utility of all primary conditions'),
           ('nih-phase2-primary-contrast.png', 'Paired interaction and uncertainty'),
           ('nih-phase2-head-privacy-penalty.png', 'Head-size comparison, three seeds'),
           ('nih-phase2-head-costs.png', 'Measured and modeled head costs'),
           ('nih-phase2-cohort-sensitivity.png', 'Client-cohort sensitivity, three seeds'),
           ('nih-phase2-clipping-sensitivity.png', 'Clipping sensitivity, three seeds')]
for name, _ in figures:
    require(hashlib.sha256((SOURCE / 'figures' / name).read_bytes()).hexdigest() == plots['files'][name], 'Figure content changed')
    sources.append(SOURCE / 'figures' / name)
OUTPUT.mkdir(parents=True, exist_ok=True)
for file in sources:
    shutil.copy2(file, OUTPUT / file.name)
columns = [('family', 'Family'), ('head', 'Head'), ('clients', 'Clients'), ('clip', 'Clip'),
           ('condition', 'Condition'), ('rule', 'Rule'), ('privacy', 'Privacy'),
           ('qualified', 'Qualified seeds'), ('auc_mean', 'Mean AUROC'), ('accuracy_mean', 'Mean accuracy')]
rows = []
for group in groups:
    cells = []
    for key, _ in columns:
        value = f'{float(group[key]):.6f}' if key.endswith('_mean') else group[key]
        cells.append(f'<td>{html.escape(value)}</td>')
    rows.append('<tr>' + ''.join(cells) + '</tr>')
primary = verification['primary']
page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Completed NIH reference study — FedLearn</title><style>
:root{color-scheme:light}body{margin:0;background:#F6F3EE;color:#191A1C;font:16px/1.55 system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:32px 20px}h1{font-size:30px;line-height:1.2}h2{font-size:21px;margin-top:32px}p{max-width:80ch}a{color:#1C314D}section{margin:24px 0;background:white;border:1px solid #E7E2D9;border-radius:12px;padding:24px}table{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}th,td{padding:8px;text-align:left;border-bottom:1px solid #E7E2D9}thead{background:#F1EDE4}.scroll{overflow:auto}.figures{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:16px}figure{margin:0}img{max-width:100%;height:auto}figcaption{font-size:14px}code{overflow-wrap:anywhere}a:focus-visible{outline:2px solid #1C314D;outline-offset:4px}</style>
<main><a href="/simulations">Return to Simulation Lab</a><h1>Completed NIH reference study</h1>
<p>Saved results, independently reconciled on 20 October 2026. This page does not start training and is separate from live exploratory validation runs.</p>
<p><strong>Task:</strong> No Finding versus any listed finding, using frozen ResNet-18 features and simulated patient groups. These results are not a clinically qualified classifier. The follow-up disease-specific baseline has not been selected or trained.</p>
<section><h2>Verified execution</h2><p>360 of 360 runs qualified; 7,200 rounds; zero failed, timed-out or unrun cells. All 360 saved models were independently checked on validation and official test data. Primary results use ten seeds; secondary comparisons use three and are descriptive.</p>
<p>The official test has already been inspected. It is a historical evaluation, not an untouched test for further model selection.</p>'''
page += f'<p>Paired interaction D: {primary["mean"]:+.9f} AUROC; 95% interval [{primary["ci95"][0]:+.9f}, {primary["ci95"][1]:+.9f}]. This interval describes seed variation conditional on fixed cohorts; a positive interaction does not imply that private trimming has the highest absolute utility.</p>'
page += '<p>Central-DP settings describe conditional per-run research mechanisms. The retained diagnostics are not DP-safe exports; no patient-level, joint-matrix or clinical guarantee follows.</p></section><h2>Audited figures</h2><div class="figures">'
for name, description in figures:
    page += f'<figure><a href="{name}"><img loading="lazy" src="{name}" alt="{html.escape(description)}"></a><figcaption>{html.escape(description)}</figcaption></figure>'
page += '</div><section><h2>All 64 recipe groups</h2><p>Official-test endpoints, averaged across the qualified seeds in each fixed group. Groups can differ in model, clipping, privacy and attack settings; this table does not establish a universal ranking.</p><div class="scroll"><table><caption>Fixed-study final outcomes by recipe group</caption><thead><tr>'
page += ''.join(f'<th scope="col">{label}</th>' for _, label in columns) + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div></section>'
page += '<h2>Source and downloads</h2><p>Execution source: <code>' + html.escape(verification['source_commit']) + '</code>. Historical report and original evidence are preserved.</p><ul>'
for name, label in [('nih-phase2-study-v1-report.md','Study report (Markdown)'),('nih-phase2-study-v1-verification.json','Independent verification receipt'),('all_64_recipe_groups.csv','All recipe groups (CSV)'),('manifest.json','Packaged file identities')]:
    page += f'<li><a download href="{name}">{label}</a></li>'
page += '</ul><p>The original Markdown report retains repository-relative references. This page contains the packaged figures and aggregates; use the repository layout to follow the report’s protocol, supplementary tables and reproduction links.</p></main></html>\n'
(OUTPUT / 'index.html').write_text(page)
manifest = {'source_commit': verification['source_commit'], 'scope': 'Read-only historical study; no live training or new scientific result',
            'source_sha256': {str(file.relative_to(ROOT)): hashlib.sha256(file.read_bytes()).hexdigest() for file in sources},
            'files': {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in sorted(OUTPUT.iterdir()) if file.is_file() and file.name != 'manifest.json'}}
(OUTPUT / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
print(f'Packaged {len(manifest["files"])} files; all 64 group rows sourced from the audited CSV.')
