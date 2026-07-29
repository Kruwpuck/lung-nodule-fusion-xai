# Panduan Kolaborasi Paper (LaTeX via Git)

Paper ini ditulis pakai **LaTeX di dalam repo Git** — tidak pakai Overleaf.
Semua kolaborator edit di komputer masing-masing, lalu `commit` + `push`.
Gratis, ter-versioning, bisa offline, dan riwayat perubahan jelas.

- **Repo:** `https://github.com/Kruwpuck/lung-nodule-fusion-xai`
- **Kelas dokumen:** IEEEtran **conference** (`\documentclass[conference]{IEEEtran}`)
- **File utama:** `paper/main.tex`
- **Output:** `paper/build/main.pdf`
- **Pengguna Windows:** lihat bagian **[Khusus Pengguna Windows](#-khusus-pengguna-windows)** di bawah.

---

## 1. Prasyarat (sekali install)

### Git
```bash
sudo apt install git          # Linux
```

### TeX Live (compiler LaTeX)
```bash
sudo apt install texlive-latex-extra texlive-bibtex-extra texlive-fonts-recommended latexmk
```
> Windows: install [MiKTeX](https://miktex.org) + Strawberry Perl (untuk latexmk).
> macOS: install [MacTeX](https://tug.org/mactex).

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

| Aksi | Command (dari dalam `paper/`) |
|------|-------------------------------|
| Build sekali | `latexmk -pdf main.tex` |
| Auto-build tiap save | `latexmk -pdf -pvc main.tex` |
| Bersihkan file sampah | `latexmk -c` |
| Build bersih dari nol | `rm -rf build && latexmk -pdf main.tex` |

Hasil PDF selalu di `paper/build/main.pdf`. Folder `build/` **tidak** ikut di-commit
(sudah di-`.gitignore`) — tiap orang build sendiri.

### Preview PDF (tanpa aplikasi tambahan)

PDF hasil build bisa dibuka pakai **PDF viewer bawaan OS** — tidak perlu Overleaf,
VS Code, atau aplikasi pihak ketiga apa pun:

| OS | Buka PDF | Viewer bawaan |
|----|----------|---------------|
| Linux (Ubuntu) | `xdg-open build/main.pdf` | GNOME Document Viewer (Evince) |
| Windows | `start build\main.pdf` | Microsoft Edge |
| macOS | `open build/main.pdf` | Preview.app |

**Preview yang auto-refresh saat mengetik** — cukup `latexmk` bawaan TeX Live,
tetap tanpa pihak ketiga:
```bash
latexmk -pdf -pvc main.tex
```
`-pvc` = *preview continuously*: latexmk membuka PDF di viewer default dan
**rebuild otomatis tiap kali `main.tex` disave**. Di Linux (Evince) PDF-nya bahkan
ikut reload sendiri. Stop dengan `Ctrl+C`.

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

> **Penting:** file `main.tex` sekarang punya baris `\nocite{*}` supaya template
> ter-compile walau belum ada sitasi. **Hapus `\nocite{*}`** begitu kamu mulai
> pakai `\cite{...}` sungguhan, supaya daftar pustaka hanya berisi yang dikutip.

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

---

## 🪟 Khusus Pengguna Windows

> **NOTICE — bagian ini KHUSUS Windows.** Pakai Linux/macOS? Abaikan.
> Alur Git & `latexmk` **sama persis**; yang beda cuma cara install & command shell.

### Install
- **Git for Windows** — <https://git-scm.com/download/win> (default cukup). Dapat
  **Git Bash** (terminal ala Linux) — disarankan dipakai untuk semua command di atas.
- **MiKTeX** (compiler LaTeX) — <https://miktex.org/download> → buka *MiKTeX Console*
  → *Update now* → Settings *"Install missing packages" = Yes*. Sudah termasuk
  `latexmk` + Perl (tak perlu install Perl terpisah).
  Alternatif: TeX Live for Windows. Pilih **salah satu**.
- **VS Code** + extension **LaTeX Workshop** (opsional, buat auto-build & preview).

### Rapikan line-ending (sekali)
Windows pakai CRLF, Linux LF — biar tidak muncul "perubahan palsu" di Git:
```bash
git config --global core.autocrlf true
```

### Command yang beda
| Aksi | Windows |
|------|---------|
| Build bersih (Git Bash) | `rm -rf build && latexmk -pdf main.tex` |
| Build bersih (PowerShell) | `Remove-Item -Recurse -Force build; latexmk -pdf main.tex` |
| Buka/preview PDF | `start build\main.pdf` (Microsoft Edge, bawaan Windows) |

Sisanya identik: `git pull/commit/push`, `latexmk -pdf main.tex`, dev mode
`latexmk -pdf -pvc main.tex`.

### Masalah umum di Windows
| Gejala | Solusi |
|--------|--------|
| `latexmk: command not found` | MiKTeX belum ke-PATH. Restart terminal, atau reinstall (centang "add to PATH"). |
| Popup MiKTeX minta install paket saat build | Klik *Install* (sekali per paket). Set "Install missing packages = Yes" biar otomatis. |
| Muncul banyak perubahan padahal tak edit | Line-ending. Jalankan `git config --global core.autocrlf true`. |
| `Permission denied` saat push | Login GitHub via browser saat diminta, atau pakai Personal Access Token sebagai password. |
| PDF tidak update di VS Code | `Ctrl+Alt+V` buka preview, atau klik 🔄 di tab PDF. |
