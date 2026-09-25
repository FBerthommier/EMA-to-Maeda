# Finalization checklist — GitHub ↔ Zenodo ↔ arXiv

The article (Interspeech 2026 submission 2479) is deposited on arXiv **after**
the GitHub and Zenodo repositories are created, so the cross-references are
unknown at preparation time. Perform these edits once, in order, when the
identifiers are known.

## Step 1 — Publish Zenodo first

- [ ] Upload the Zenodo package (see `Zenodo/GUIDE.md`).
- [ ] Publish and obtain the **Zenodo DOI** (concept DOI preferred).
- [ ] In Zenodo metadata, add a *related identifier* "isSupplementTo" → GitHub
      repository URL.

## Step 2 — Update GitHub

- [ ] `README.md`: replace every `PLACEHOLDER` in the finalization banner and
      the DOI badge with the Zenodo DOI; insert the arXiv identifier
      (`arXiv:XXXX.XXXXX`) and the final repository URL in the clone command.
- [ ] `CITATION.cff`: replace `doi` with the Zenodo DOI; add the `preferred-citation`
      entry with the arXiv identifier and the official Interspeech-style
      reference.
- [ ] `docs/documentation.tex` / `.pdf`: update the banner and references,
      recompile the PDF.
- [ ] `video/README.md` and `docs/mapping.md`: check for remaining `PLACEHOLDER`
      strings (`grep -r PLACEHOLDER .`).

## Step 3 — Update the article (arXiv)

- [ ] Add the footnote/acknowledgment block per the official Interspeech model:
      "Code and data available at https://github.com/FBerthommier/EMA-to-Maeda
      (MIT license); complete archive including supplement, figures, videos and
      simulations archived at Zenodo <DOI>."
- [ ] Add the Zenodo DOI and GitHub URL to the article's *Data availability*
      section if applicable.
- [ ] Submit to arXiv; obtain the arXiv identifier.
- [ ] Back-link: add "isIdenticalTo → arXiv:XXXX.XXXXX" as a related identifier
      in Zenodo; mention the arXiv ID in the GitHub README banner.

## Step 4 — Final consistency pass

- [ ] All three repositories point to each other with valid, resolvable URLs.
- [ ] Zenodo version released and DOI resolves to the published deposit.
- [ ] Fresh clone → `pip install -r requirements.txt` → one figure script runs
      end-to-end without errors.
