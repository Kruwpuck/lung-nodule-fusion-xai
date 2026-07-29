# Panduan Kolaborasi Paper (LaTeX via Git)

Paper ini ditulis pakai **LaTeX di dalam repo Git** — tidak pakai Overleaf.
Semua kolaborator edit di komputer masing-masing, lalu `commit` + `push`.
Gratis, ter-versioning, bisa offline, dan riwayat perubahan jelas.

- **Repo:** `https://github.com/Kruwpuck/lung-nodule-fusion-xai`
- **Kelas dokumen:** IEEEtran **conference** (`\documentclass[conference]{IEEEtran}`)
- **File utama:** `paper/main.tex`
- **Output:** `paper/build/main.pdf`

---

## 1. Prasyarat (sekali install)

### Git

**Linux:**
```bash
sudo apt install git
```

**Windows:** Install [Git for Windows](https://git-scm.com/download/win) (default cukup).
Dapat **Git Bash** — pakai Git Bash untuk semua command di panduan ini.

**macOS:** Git sudah tersedia, atau `brew install git`.

---

**Windows — rapikan line-ending (sekali, setelah install Git):**
```bash
git config --global core.autocrlf true
```
Cegah "perubahan palsu" karena beda format baris Windows (CRLF) vs Linux/macOS (LF).

### LaTeX Compiler

**Linux:**
```bash
sudo apt install texlive-latex-extra texlive-bibtex-extra texlive-fonts-recommended latexmk
```

**Windows:** Install [MiKTeX](https://miktex.org/download) → buka *MiKTeX Console*
→ *Update now* → Settings *"Install missing packages" = Yes*.
Sudah termasuk `latexmk` + Perl. Alternatif: TeX Live for Windows — pilih **salah satu**.

**macOS:** Install [MacTeX](https://tug.org/mactex).

### Editor (opsional tapi enak)
- **VS Code** + extension **LaTeX Workshop** → auto-build tiap save, preview PDF di samping.
- Atau editor apa saja + `latexmk` manual.

---

## 2. Setup Awal

```bash
git clone https://github.com/Kruwpuck/lung-nodule-fusion-xai.git
cd lung-nodule-fusion-xai/paper
latexmk -pdf main.tex        # build pertama → build/main.pdf
```

Semua file yang dibutuhkan (`IEEEtran.cls`, `IEEEtran.bst`, `refs.bib`) sudah ada
di repo. Tidak perlu download template lagi.

---

## 3. Cara Build

| Aksi | Linux / macOS / Git Bash | PowerShell (Windows) |
|------|--------------------------|----------------------|
| Build sekali | `latexmk -pdf main.tex` | sama |
| Auto-build tiap save | `latexmk -pdf -pvc main.tex` | sama |
| Bersihkan file sampah | `latexmk -c` | sama |
| Build bersih dari nol | `rm -rf build && latexmk -pdf main.tex` | `Remove-Item -Recurse -Force build; latexmk -pdf main.tex` |

Hasil PDF selalu di `paper/build/main.pdf`. Folder `build/` **tidak** ikut di-commit
(sudah di-`.gitignore`) — tiap orang build sendiri.

### Preview PDF

| OS | Command | Viewer bawaan |
|----|---------|---------------|
| Linux (Ubuntu) | `xdg-open build/main.pdf` | GNOME Document Viewer (Evince) |
| Windows | `start build\main.pdf` | Microsoft Edge |
| macOS | `open build/main.pdf` | Preview.app |

**Auto-refresh saat mengetik:**
```bash
latexmk -pdf -pvc main.tex
```
`-pvc` rebuild otomatis tiap `main.tex` disave. Di Linux (Evince) PDF-nya reload sendiri.
Stop dengan `Ctrl+C`.

---

## 4. Alur Edit Harian (WAJIB diikuti biar tidak bentrok)

```
1. git pull                 ← SELALU tarik update terbaru dulu
2. edit main.tex            ← tulis bagianmu
3. latexmk -pdf main.tex    ← pastikan TIDAK ada error
4. git add -A
5. git commit -m "pesan jelas, mis: tulis subsection Methodology"
6. git pull                 ← tarik lagi kalau-kalau ada yang push barusan
7. git push                 ← kirim
```

Kalau langkah 7 ditolak (`rejected`), berarti ada yang push duluan:
```bash
git pull --rebase
# selesaikan konflik kalau ada (lihat bagian 6), lalu:
git push
```

---

## 5. Menambah Referensi / Sitasi

Bibliografi ada di **`paper/refs.bib`**. Ada dua cara:

### Cara A — Zotero (disarankan, tidak edit tangan)
1. Simpan paper ke **Zotero** (via Connector browser atau Add by Identifier DOI/arXiv).
2. Zotero + plugin **Better BibTeX** meng-*auto-export* ke `paper/refs.bib`
   (fitur "Keep Updated").
3. Citekey dibuat otomatis, format: `auth.lower + shorttitle(3,3) + year`
   (mis. `armatoLungImage2011`).
4. `commit` + `push` `refs.bib` yang sudah ter-update.

> Sepakati **satu orang** sebagai pengelola `refs.bib` via Zotero, atau koordinasi
> saat export, supaya tidak saling menimpa.

### Cara B — Manual (edit refs.bib langsung)
Tambah entri BibTeX baru, pastikan **citekey unik**:
```bibtex
@article{armatoLungImage2011,
  author  = {Armato, Samuel G. and others},
  title   = {The Lung Image Database Consortium (LIDC)},
  journal = {Medical Physics},
  year    = {2011},
  volume  = {38},
  number  = {2},
  pages   = {915--931}
}
```

### Mengutip di `main.tex`
```latex
Metode ini mengikuti \cite{armatoLungImage2011}.
Beberapa sekaligus: \cite{keyA, keyB}.
```

> **Catatan:** `main.tex` sekarang pakai daftar referensi bawaan template (b1–b11)
> supaya langsung ter-compile. Begitu mulai pakai referensi asli via Zotero, hapus
> blok `\begin{thebibliography}...\end{thebibliography}` dan aktifkan dua baris ini:
> ```latex
> \bibliographystyle{IEEEtran}
> \bibliography{refs}
> ```
> Petunjuk lengkap ada di komentar di bawah `\end{document}` di `main.tex`.

---

## 6. Menyelesaikan Konflik (conflict)

Kalau dua orang mengedit baris yang sama, `git pull` akan menandai:
```
<<<<<<< HEAD
teksmu
=======
teks orang lain
>>>>>>> abc123
```
Edit manual: pilih/gabung yang benar, hapus ketiga baris penanda, lalu:
```bash
latexmk -pdf main.tex     # cek tetap ter-build
git add main.tex
git commit
git push
```

**Cara paling ampuh menghindari konflik:** bagi tugas per-*section*. Kalau paper
makin besar, pecah jadi beberapa file dan `\input{}` di `main.tex`:
```latex
\input{sections/methodology.tex}
```
Tiap orang pegang file section berbeda → nyaris tak pernah bentrok.

---

## 7. File Penting

| File | Fungsi | Boleh diedit? |
|------|--------|---------------|
| `main.tex` | Isi paper | ✅ ya |
| `refs.bib` | Daftar referensi | ✅ ya (atau via Zotero) |
| `figures/` | Gambar (.png/.pdf) | ✅ tambah gambar di sini |
| `IEEEtran.cls` | Kelas dokumen IEEE | ❌ jangan |
| `IEEEtran.bst` | Style bibliografi IEEE | ❌ jangan |
| `.latexmkrc` | Config build | ❌ jarang perlu |
| `build/` | Output (PDF, dll) | ❌ tidak di-commit |

---

## 8. Aturan Singkat

- **Selalu `git pull` sebelum mulai dan sebelum push.**
- **Commit kecil & sering**, pesan yang jelas.
- **Jangan commit `build/`** (sudah otomatis di-abaikan).
- **Jangan edit `IEEEtran.cls` / `IEEEtran.bst`.**
- Gambar taruh di `figures/`, panggil `\includegraphics{namafile}`.
- Kalau ragu, build dulu — jangan push kalau masih error.

### Masalah Umum

| Gejala | Solusi |
|--------|--------|
| `latexmk: command not found` (Windows) | MiKTeX belum ke-PATH. Restart terminal, atau reinstall (centang "add to PATH"). |
| Popup MiKTeX minta install paket | Klik *Install* (sekali per paket). Set "Install missing packages = Yes" biar otomatis. |
| Banyak perubahan padahal tak edit (Windows) | Line-ending. Jalankan `git config --global core.autocrlf true`. |
| `Permission denied` saat push | Login GitHub via browser saat diminta, atau pakai Personal Access Token sebagai password. |
| PDF tidak update di VS Code | `Ctrl+Alt+V` buka preview, atau klik 🔄 di tab PDF. |
