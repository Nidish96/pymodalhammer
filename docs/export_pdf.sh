#!/bin/sh
# Build pymodalhammer_manual.pdf from the Org source. Needs Emacs (Org ≥ 9.4), XeLaTeX and the DejaVu
# fonts. Same as C-c C-e l p in Emacs.
cd "$(dirname "$0")" && emacs --batch pymodalhammer_manual.org \
    --eval "(setq org-latex-pdf-process '(\"xelatex -interaction nonstopmode -output-directory %o %f\" \"xelatex -interaction nonstopmode -output-directory %o %f\" \"xelatex -interaction nonstopmode -output-directory %o %f\"))" \
    --eval "(setq org-latex-remove-logfiles nil)" \
    -f org-latex-export-to-pdf
