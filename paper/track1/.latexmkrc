$pdf_mode = 1;
$bibtex_use = 1;
$out_dir = 'build';
@generated_exts = (@generated_exts, 'synctex.gz', 'bbl', 'blg');
# Jalur mutlak, bukan '..'. bibtex dijalankan latexmk dari dalam $out_dir sehingga
# jalur relatif teruraikan dari direktori yang berbeda dengan pdflatex; akibatnya
# bibtex melapor "I couldn't open database file refs.bib", thebibliography terbit
# kosong, dan build gagal dengan "missing \item" -- padahal sitasinya benar.
use Cwd ();
my $paper_dir = Cwd::abs_path('..');
ensure_path('TEXINPUTS', $paper_dir);
ensure_path('BIBINPUTS', $paper_dir);
