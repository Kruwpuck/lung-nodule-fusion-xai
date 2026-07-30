$pdf_mode = 1;
$bibtex_use = 1;
$out_dir = 'build';
@generated_exts = (@generated_exts, 'synctex.gz', 'bbl', 'blg');
ensure_path('TEXINPUTS', '..');
ensure_path('BIBINPUTS', '..');
