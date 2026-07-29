# Panduan Kolaborasi Paper — Versi Windows

Versi Windows dari [`PANDUAN-KOLABORASI.md`](PANDUAN-KOLABORASI.md). Alur Git &
`latexmk` **sama persis**; yang beda cuma cara install dan command shell.

- **Repo:** `https://github.com/Kruwpuck/lung-nodule-fusion-xai`
- **Kelas dokumen:** IEEEtran **conference**
- **File utama:** `paper\main.tex` → **Output:** `paper\build\main.pdf`

---

## 1. Install (sekali)

### Git for Windows
Download: <https://git-scm.com/download/win> → install (default sudah cukup).
Ini sekaligus memberi **Git Bash** (terminal ala Linux) — disarankan dipakai.

### MiKTeX (compiler LaTeX) — cara termudah di Windows
1. Download: <https://miktex.org/download> → install.
2. Jalankan **MiKTeX Console** → tab *Updates* → *Update now*.
3. Setelan *Settings* → *"Install missing packages"* set ke **Yes** (atau *Ask*),
   biar paket LaTeX yang kurang ke-download otomatis saat build.
4. MiKTeX sudah termasuk `latexmk` + Perl — tidak perlu install Perl terpisah.

> Alternatif: **TeX Live for Windows** (<https://tug.org/texlive>) — lebih besar
> tapi lengkap. Pilih salah satu, jangan dua-duanya.

### Editor — VS Code (disarankan)
1. Download VS Code: <https://code.visualstudio.com>
2. Install extension **LaTeX Workshop** (James Yu).
3. Buka folder `paper\` → LaTeX Workshop auto-build tiap save, preview PDF di samping.

---

## 2. Setup Awal

Buka **Git Bash** (klik kanan di folder → *Git Bash Here*) atau **PowerShell**:
```bash
git clone https://github.com/Kruwpuck/lung-nodule-fusion-xai.git
cd lung-nodule-fusion-xai/paper
latexmk -pdf main.tex
```
Semua file (`IEEEtran.cls`, `IEEEtran.bst`, `refs.bib`) sudah ada di repo.

### Rapikan line-ending (sekali, hindari "false changes")
Windows pakai CRLF, Linux LF. Biar tidak muncul perubahan palsu di Git:
```bash
git config --global core.autocrlf true
```

---

## 3. Cara Build

| Aksi | Command (dari dalam `paper\`) |
|------|-------------------------------|
| Build sekali | `latexmk -pdf main.tex` |
| Auto-build tiap save | `latexmk -pdf -pvc main.tex` |
| Bersihkan file sampah | `latexmk -c` |
| Build bersih (Git Bash) | `rm -rf build && latexmk -pdf main.tex` |
| Build bersih (PowerShell) | `Remove-Item -Recurse -Force build; latexmk -pdf main.tex` |

Pakai **VS Code + LaTeX Workshop** lebih gampang: cukup save (`Ctrl+S`), PDF
langsung ter-update. Tombol build ada di ikon ▶ (TeX) sidebar kiri.

Hasil PDF: `paper\build\main.pdf`. Folder `build\` **tidak** di-commit.

---

## 4. Alur Edit Harian (WAJIB, biar tidak bentrok)

Command Git identik di semua OS:
```bash
git pull                    # SELALU tarik update dulu
# ...edit main.tex, build, pastikan tidak error...
git add -A
git commit -m "pesan jelas"
git pull                    # tarik lagi kalau ada yang push barusan
git push
```
Kalau `push` ditolak (`rejected`):
```bash
git pull --rebase
# selesaikan konflik kalau ada, lalu:
git push
```

---

## 5. Referensi, Konflik, Aturan

Untuk **menambah referensi (`refs.bib`)**, **menyelesaikan konflik**, **daftar
file penting**, dan **aturan singkat** — sama persis dengan versi utama.
Baca: [`PANDUAN-KOLABORASI.md`](PANDUAN-KOLABORASI.md) bagian 5–8.

Ringkas:
- Tambah referensi via **Zotero + Better BibTeX** (auto-export ke `refs.bib`), atau
  edit `refs.bib` manual (citekey unik) → `\cite{citekey}` di `main.tex`.
- **Hapus `\nocite{*}`** di `main.tex` begitu mulai pakai `\cite{}` sungguhan.
- **Jangan** edit `IEEEtran.cls` / `IEEEtran.bst`. Gambar taruh di `figures\`.
- **Selalu `git pull` sebelum mulai & sebelum push.**

---

## 6. Masalah Umum di Windows

| Gejala | Solusi |
|--------|--------|
| `latexmk: command not found` | MiKTeX/TeX Live belum ke-PATH. Restart terminal, atau reinstall MiKTeX (centang "add to PATH"). |
| Build minta install paket, muncul popup MiKTeX | Klik *Install* (sekali per paket). Set "Install missing packages = Yes" biar otomatis. |
| Muncul banyak perubahan padahal tidak edit | Line-ending. Jalankan `git config --global core.autocrlf true`. |
| `Permission denied` saat push | Login GitHub via browser saat diminta, atau pakai Personal Access Token sebagai password. |
| PDF tidak ter-update di VS Code | `Ctrl+Alt+V` buka preview, atau klik ikon 🔄 di tab PDF. |
