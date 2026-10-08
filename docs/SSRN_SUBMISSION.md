# Submitting this paper to SSRN

The manuscript is `paper/main.pdf` (compiled from `paper/main.tex` with
Tectonic or pdfLaTeX). SSRN accepts a PDF upload; the LaTeX sources in this
repository are provided so the paper can be revised and re-compiled.

## 1. Compile the final version

```bash
python scripts/10_statistics.py     # regenerates statistics + inlines tables
python scripts/11_paper_numbers.py  # refreshes every number in the manuscript
cd paper && tectonic main.tex       # or: pdflatex main.tex
```

Verify that no number reads `n/a` (that would mean an experiment has not been
run yet):

```bash
grep -n "n/a" paper/main.tex | head
```

## 2. Prepare the submission package

* `paper/main.pdf` — the manuscript (anonymous or with author names, see below)
* Cover sheet / title page with:
  * full title
  * author names, affiliations, emails
  * corresponding author
  * abstract (already in the manuscript; copy it into the SSRN abstract field)
  * **keywords**: optogenetics, channelrhodopsin, computational neuroscience,
    neural encoding models, deep learning, spiking neurons, distribution shift,
    predictive uncertainty, closed-loop stimulation design
  * **JEL classification**: C45, C63, I23, Q05
  * JEL classification codes are optional on SSRN but help reviewers
* Optional: the code repository link, and the statement that all code and the
  data generator are open (MIT).

## 3. Create the SSRN submission

1. Go to <https://www.ssrn.com> and create an author account (or sign in).
2. **Submit a paper → Add a new submission → Working Paper** (this is the free
   route; do not pick "Journal article" unless you have a specific journal
   agreement).
3. Upload `paper/main.pdf` as the manuscript.
4. Fill in the metadata fields above.
5. Under **Disclosures**, state any funding and any AI-assistance disclosure
   required by your institution. SSRN requires an AI disclosure to appear
   **both** in the abstract metadata and **on the PDF itself** (the manuscript
   carries one directly under the abstract); update that sentence if your
   usage differed from what is printed there.
6. Choose the **subject area** closest to: *Computer Science and Law →
   Computers and Information* or *Econometrics → Statistics and Econometrics
   Methodology*, and optionally *Neuroscience → Computational Neuroscience*.
7. **Keywords**: paste the list above.
8. Confirm and post. SSRN assigns a permanent DOI within ~1 business day.

## 4. Making the GitHub companion

```bash
git init
git add .
git commit -m "OptoNet: benchmark, models, experiments and manuscript"
gh repo create optonet --public --source . --push       # or create the repo on github.com first
```

Useful additions before you post:

* add `YOUR-USERNAME` to `CITATION.cff` and the repository URL in the paper's
  Data-and-code section,
* add a `CITATION.cff`-aware badge in the README,
* consider adding a `Makefile`/GitHub Action badge for the test suite,
* write a short Twitter/X thread: the headline is *"trees fail on new
  stimulation patterns, sequence models don't"* — that contrast is the
  shareable part.

## 5. What reviewers will ask

Prepare answers for these before posting; all of them are answered in the
manuscript or the repository:

| Question | Where it is answered |
|---|---|
| Is this only a simulation study? | `Limitations` §6, and `docs/METHODS.md` |
| Would it work on real data? | `docs/REAL_DATA.md` |
| Is the data leakage-free? | cells are split by *cell*, never by trial; see `src/optonet/data.py` |
| Are the models causal? | `tests/test_models.py::test_no_future_leakage_beyond_group` |
| Is the comparison fair? | identical loss, splits, optimiser and metrics; parameter counts reported |
| Why not a bigger model? | `Limitations` §6 — data/parameter trade-off is reported explicitly |
