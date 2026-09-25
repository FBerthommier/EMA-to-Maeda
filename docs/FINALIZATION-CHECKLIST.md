# Finalization checklist — GitHub ↔ Zenodo ↔ arXiv

The article (Interspeech 2026 submission 2479) is deposited on arXiv **after**
the GitHub and Zenodo repositories are created, so the cross-references are
unknown at preparation time. Perform these edits once, in order, when the
identifiers are known.

## Step 1 — Publish Zenodo first

- [x] Upload the Zenodo package (see `Zenodo/GUIDE.md`).
- [x] Publish and obtain the **Zenodo DOI**: `10.5281/zenodo.22961484`.
- [ ] In Zenodo metadata, add a *related identifier* "isSupplementTo" →
      https://github.com/FBerthommier/EMA-to-Maeda

## Step 2 — Update GitHub

- [x] `README.md`: DOI badge, finalization banner and citation BibTeX carry
      the Zenodo DOI `10.5281/zenodo.22961484`; final repository URL
      https://github.com/FBerthommier/EMA-to-Maeda in the banner and clone
      command. (Remaining: the arXiv identifier once submitted.)
- [x] `CITATION.cff`: `doi` set to the Zenodo DOI; `repository-code` points to
      GitHub. (Remaining: `preferred-citation` with the arXiv identifier and
      the official Interspeech-style reference.)
- [x] `docs/documentation.tex` / `.pdf` and `docs/usage_manual.tex` / `.pdf`:
      banner carries the GitHub URL and the Zenodo DOI, PDFs recompiled.
      (Remaining: the arXiv identifier.)
- [x] No remaining `PLACEHOLDER` strings except the arXiv identifier itself
      (`grep -r PLACEHOLDER .`).

## Step 3 — Update the article (arXiv)

- [ ] Add the footnote/acknowledgment block per the official Interspeech model:
      "Code and data available at https://github.com/FBerthommier/EMA-to-Maeda
      (MIT license); complete archive including supplement, figures, videos and
      simulations archived at Zenodo, doi:10.5281/zenodo.22961484."
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
