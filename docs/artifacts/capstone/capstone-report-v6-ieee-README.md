# IEEE capstone report

28 October 2026. This manuscript integrates the frozen medical comparison, three-host CPU measurements and completed generated projection study. Earlier reports and scientific evidence remain unchanged.

The structure follows the supplied photograph: title and affiliation; abstract; Introduction; Solution with overview and components; Details; Evaluation with Experimental Setup and Results; Discussion; Related Work; Conclusion and Future Work; References. IEEEtran conference mode supplies the two-column layout, Roman-numbered sections, lettered subsections, typography and numbered citations.

## Files

- `capstone-report-v6-ieee.tex`: editable manuscript.
- `capstone-report-v6-ieee.pdf`: nine-page reading copy.
- `figures/ieee-2026-10-28/*.pdf`: six vector plots redrawn from frozen aggregate records for column-sized labels. The architecture diagram is embedded directly in the manuscript.
- `repro/render_ieee_figures.py`: optional figure reproduction from the three recorded public JSON inputs.
- `capstone-report-v6-ieee-verification.json`: source/input bindings, numerical checks and visual inspection record.

The delivery directory also contains `capstone-report-v6-ieee-source.zip`, with the manuscript and every figure needed to compile it. It does not contain raw medical data, model files or the reference photograph.

## Compilation

Use a current TeX distribution with IEEEtran, or upload the source ZIP to a LaTeX editor with IEEEtran support. The manuscript uses standard packages: fontenc, cite, amsmath, amssymb, graphicx, booktabs, array, url, hyperref, TikZ and balance. No bibliography database is required; references are included in the source.

```sh
latexmk -pdf capstone-report-v6-ieee.tex
```

Alternatively:

```sh
tectonic capstone-report-v6-ieee.tex
```

Run from the directory containing the manuscript and `figures/`. The reading copy uses US Letter paper, IEEEtran conference defaults and embedded fonts. Layout work does not change the original estimands, uncertainty targets or study results.
