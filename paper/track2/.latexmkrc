$pdf_mode = 1;
# 2, bukan 1. Pada 1 latexmk lebih dulu mencari sendiri refs.bib lewat kpsewhich,
# gagal menemukannya, lalu MEMVETO bibtex tanpa menjadikannya error. Build tetap
# "sukses" memakai .bbl basi. Catatan lengkapnya di paper/track1/.latexmkrc.
$bibtex_use = 2;
$out_dir = 'build';
@generated_exts = (@generated_exts, 'synctex.gz', 'bbl', 'blg');
# Jalur mutlak, bukan '..'. bibtex dijalankan latexmk dari dalam $out_dir sehingga
# jalur relatif teruraikan dari direktori yang berbeda dengan pdflatex; akibatnya
# bibtex melapor "I couldn't open database file refs.bib", thebibliography terbit
# kosong, dan build gagal dengan "missing \item" -- padahal sitasinya benar.
use Cwd ();
my $paper_dir = Cwd::abs_path('..');
# Dijalankan dari Git Bash, perl mengembalikan jalur gaya MSYS ('/c/Users/...'),
# yang tidak dimengerti MiKTeX. Itu sebab sebenarnya bug BIBINPUTS, bukan jalur
# relatifnya. Track 2 belum punya \bibliography, jadi perangkapnya belum menggigit
# di sini -- diperbaiki sekarang supaya tidak menunggu sitasi pertama.
$paper_dir =~ s{^/([A-Za-z])/}{\u$1:/};
ensure_path('TEXINPUTS', $paper_dir);
ensure_path('BIBINPUTS', $paper_dir);
