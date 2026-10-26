# paper/track1 and paper/track2 are mirrors

The canonical manuscripts are `drafting/track1-main.tex` and `drafting/track2-main.tex` (outside this repository, in the
authors' working folder). `paper/track1/main.tex`, `paper/track2/main.tex` and the `images/` folders next to them are copies
made by `drafting/sync_to_paper.py` so that collaborators can compile from git. Do not edit them here; send changes to the
author who keeps `drafting/`. The copies replace the older drafts that were here before (see git history).

Each manuscript carries its bibliography inline (`thebibliography`), so `refs.bib` is not used by these two files.
Build: `cd paper/track1 && latexmk -pdf main.tex` (same for `track2`).
