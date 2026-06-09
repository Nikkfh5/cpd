# Release Notes: v1.0

This release prepares the public repository as the reproducibility artifact for
the FNL manuscript on change point detection for generated stochastic
trajectories with level-crossing labels.

## Included

- Public English README and reproducibility guide.
- Data-generation policy checks and public generation contract.
- Frozen benchmark outputs used by the manuscript tables.
- Frozen same-noise diagnostic outputs.
- Table and figure builders for manuscript-facing artifacts.
- FNL manuscript source and generated article assets under `manuscript/`.
- Citation metadata and MIT license.

## Verification

Run:

```bash
python smoke_test.py
python scripts/build_publication_tables.py
python scripts/build_same_noise_tables.py
python scripts/build_article_compact_tables.py
python scripts/build_publication_figures.py
```

Build the manuscript PDF:

```bash
cd manuscript/src
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Archival

No DOI is claimed in this release note until an archival record exists. After
the GitHub release is created, the release archive can be deposited through
Zenodo or an equivalent archive and the DOI can be added to the manuscript and
repository metadata.
