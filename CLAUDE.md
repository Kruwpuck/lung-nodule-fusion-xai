Stack: LaTeX IEEEtran conference (bibtex). Build: cd paper && latexmk -pdf main.tex
Engine: ~/research-engine (search.py, ingest.py, convert.py)
Vault: ~/research-vault (READ-ONLY research context; write only when asked to make a note)
Bib: paper/refs.bib (Zotero auto-export Better BibTeX — do not hand-edit)
Rules:
- Never invent citekeys. If key missing from refs.bib, say so.
- Write .tex in paper/ only. Do not edit vault.
- Run latexmk after every edit; report errors.
- Citation style: \cite{key} — bibtex + IEEEtran.bst
