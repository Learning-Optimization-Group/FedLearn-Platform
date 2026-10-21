# Reproduce the working report PDF

`render_working_report.py` renders the current historical-study manuscript and
page-level PNGs. It does not train, score, select models or modify study evidence.
The source Markdown, referenced figures, explicit report date, four font files and
pinned rendering dependencies determine the output. The checked 21 October build
contains eight pages and includes the separately verified simulator checkpoint.
Advisor-approved submission format/title, the disease follow-up and final presentation
remain outside this working-render acceptance.

Use Python 3.12 and an isolated environment, then install the tested versions in
`requirements-render.txt`. Supply a directory containing these DejaVu Serif files:

- `DejaVuSerif.ttf`
- `DejaVuSerif-Bold.ttf`
- `DejaVuSerif-Italic.ttf`
- `DejaVuSerif-BoldItalic.ttf`

The layout receipt records all four font hashes; match those bytes when reproducing
this exact layout. Fonts are a declared input, rather than an assumed machine path.
From the repository root, with `CAPSTONE_FONT_DIR` set to that directory:

```sh
python docs/artifacts/capstone/repro/render_working_report.py \
  --source docs/artifacts/capstone/capstone-final-report-v2-working.md \
  --output .temp/report-preview/report.pdf \
  --pages .temp/report-preview/pages \
  --font-dir "$CAPSTONE_FONT_DIR" \
  --report-date 2026-10-21
```

Use fresh output and page directories: the command refuses to overwrite them.
It retains a render receipt binding source, renderer, PDF, referenced figures,
font files and dependency versions, plus the page count and extracted text checks.
The date is explicit; the old scratch renderer hard-coded 20 October and is no
longer the current rendering entry point. Fixed PDF metadata removes clock-dependent
changes. Two fresh renders with the checked inputs/environment were byte-identical;
this is not a claim about arbitrary fonts, library versions or operating systems.

Inspect every generated page before replacing a delivered report. Text extraction
checks key numerical strings and replacement glyphs, but cannot establish layout
quality or scientific validity. The checked delivery additionally compared methods
and results text with the verified prior manuscript, confirmed unchanged study and
figure identities, inspected all eight page images, and recorded their hashes.
The public `capstone-final-report-v2-working-layout.json` records that acceptance.

The earlier report remains available in the verified 20 October delivery commit,
`4e8003320d9dddc9db0bf08a53d2825a9e8ef63d`. Its scientific study artifacts are
preserved. The current working report and layout receipt supersede only that report
preview; the frozen protocol, raw results, audited tables/figures and historical
study report retain their original identities.
