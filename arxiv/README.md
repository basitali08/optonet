# arXiv submission package

Upload file: **`optonet-arxiv.tar.gz`** (631 KB) — contains `main.tex`,
`main.bbl`, `refs.bib` and all 9 figures. Verified: compiles standalone,
13 pages, all figures/tables/references resolve, `n/a` count = 0.

## How to submit (≈20 minutes)

1. Create an account: https://arxiv.org/login → sign up (your email).
2. New submission: https://arxiv.org/auth/submit → paste metadata below.
3. Upload `optonet-arxiv.tar.gz` → press "Preview" to verify it compiles.
4. Review → approve → submission is announced within ~1 business day
   (moderation for new submitters can take 1–2 days more).

## Metadata to enter

**Title**

```
OptoNet: Deep Learning Predicts and Designs Neuron-Type-Specific Responses to Optogenetic Stimulation
```

**Abstract** — copy verbatim from `paper/main.tex` between
`\begin{abstract}` and `\end{abstract}` (strip LaTeX commands: `\opto{}` →
OptoNet, `\former{}` → OptoFormer, `$\Rtwo$` → R^2, `\emph{}` → italics,
macros like `\RtwoTransformerId` → their numbers — easiest source: paste the
abstract text from the published `paper/main.pdf`).

**Comments**

```
13 pages, 9 figures, 2 tables. Code: https://github.com/basitali08/optonet
```

**Subjects**

- Primary: **cs.LG** (Machine Learning)
- Cross-lists: **q-bio.NC** (Neurons and Cognition), **eess.SP** (Signal Processing)

**License**: arXiv.org perpetual non-exclusive license (default) — fine.

## First-time submitter: endorsement

cs.LG requires an **endorsement** the first time you submit: someone who has
published in cs.LG in the last year must endorse you (a professor who knows
your work — e.g., a recommender for your applications — can register and
endorse you in minutes; endorsement is about category competence, not paper
quality).

If you cannot find an endorser:
- arXiv's help form can assist: https://info.arxiv.org/help/endorsement/
- Or post the same package to **bioRxiv** (computational biology, no
  endorsement needed) and/or keep SSRN as the citable preprint.

## After acceptance

- Cite the arXiv ID in the GitHub README / CITATION.cff.
- Update the paper's data-availability line to include the arXiv ID.
- Link the arXiv abstract page from your CV and application materials.
