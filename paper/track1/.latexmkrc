$pdf_mode = 1;
# 2, bukan 1. Pada 1 latexmk lebih dulu mencari sendiri refs.bib lewat kpsewhich,
# gagal menemukannya, lalu MEMVETO bibtex tanpa menjadikannya error -- padahal
# BIBINPUTS di bawah sudah benar dan bibtex sendiri berhasil membaca berkasnya.
# Build tetap "sukses" memakai .bbl basi, jadi sitasi baru tidak pernah muncul dan
# tidak ada yang mengeluh. Persis bentuk kegagalan senyap yang dicatat sec 8.10.
# Pada 2 bibtex selalu dijalankan, sehingga kesalahan jalur gagal dengan berisik.
$bibtex_use = 2;
$out_dir = 'build';
@generated_exts = (@generated_exts, 'synctex.gz', 'bbl', 'blg');
# Jalur mutlak, bukan '..'. bibtex dijalankan latexmk dari dalam $out_dir sehingga
# jalur relatif teruraikan dari direktori yang berbeda dengan pdflatex; akibatnya
# bibtex melapor "I couldn't open database file refs.bib", thebibliography terbit
# kosong, dan build gagal dengan "missing \item" -- padahal sitasinya benar.
use Cwd ();
my $paper_dir = Cwd::abs_path('..');
# Dijalankan dari Git Bash, perl mengembalikan jalur gaya MSYS ('/c/Users/...').
# MiKTeX tidak mengerti bentuk itu: kpsewhich diam-diam tidak menemukan apa pun,
# dan itulah sebab sebenarnya bug BIBINPUTS -- bukan jalur relatif, melainkan
# jalur mutlak dengan gaya yang salah. Dikembalikan ke bentuk drive Windows.
$paper_dir =~ s{^/([A-Za-z])/}{\u$1:/};
ensure_path('TEXINPUTS', $paper_dir);
ensure_path('BIBINPUTS', $paper_dir);
