# PLAN-ESCALATION-LOOP

Loop semi-otomatis. Eksekutor murah + perencana mahal + gate manusia untuk research dalam.
Semua langkah dicatat. Output = artefak penelitian, bukan cuma kode jalan.

**Mode**: B — human-in-the-loop
**Target**: 1 mesin (VPS Ubuntu, akses via WireGuard), 1 CLI (Claude Code)
**Prinsip**: eksekutor tidak riset. Perencana tidak ngoding. Handoff = file `.md`.

---

## 0. SLOT ISIAN

Isi dulu. Jangan mulai sebelum semua terisi.

```
PROJECT_ROOT     = ______________________   # mis. ~/exp-arsitektur-model
GOAL             = ______________________   # 1 kalimat, punya kriteria selesai terukur
DONE_CRITERIA    = ______________________   # mis. "pytest hijau + metrik X >= 0.90"
GPU_HOST         = ______________________   # host training. kosongkan kalau tidak ada
MAX_ITER         = 15                       # batas total iterasi eksekusi
MAX_SUBAGENT     = 5                        # keras. jangan naikkan
MAX_FETCH_PER_SUB= 8                        # batas web_fetch per subagent
RUN_ID           = ______________________   # mis. 2026-08-04-run01
RESERVED_ENV     = ______________________   # venv Scrapy dsb. AGENT DILARANG SENTUH
```

---

## FASE 0 — RECON (READ-ONLY, WAJIB)

Agent **dilarang menulis apa pun** di fase ini.

- [ ] `pwd`, `git status`, `git log --oneline -10`
- [ ] Petakan struktur repo (maks kedalaman 3)
- [ ] Catat: bahasa, package manager, cara jalankan test, cara jalankan training
- [ ] Cek `claude --version`, cek skill yang terpasang: `ls ~/.claude/skills`
- [ ] Cek apakah `GPU_HOST` bisa di-SSH. Kalau tidak, catat sebagai batasan
- [ ] Tulis temuan ke layar saja

**GATE 0 — manusia baca ringkasan RECON, ketik `LANJUT`.**
Tanpa `LANJUT`, berhenti.

---

## FASE 1 — SCAFFOLD

```bash
mkdir -p $PROJECT_ROOT/{code,handoff,runs/$RUN_ID,research}
cd $PROJECT_ROOT && git init 2>/dev/null || true
```

Struktur:

```
PROJECT_ROOT/
├── code/                     # kode kerja
├── handoff/                  # kontrak antar-agent (aktif)
│   ├── GOAL.md
│   ├── STUCK.md
│   ├── BRIEF.md
│   ├── BLOCKED.md
│   └── RESEARCH.md
├── research/                 # arsip permanen, masuk vault nanti
├── runs/<RUN_ID>/            # log mentah, append-only
│   ├── events.jsonl
│   ├── ledger.md
│   └── transcripts/
└── STATE.json                # posisi loop sekarang
```

`handoff/GOAL.md` diisi manual:

```markdown
# GOAL
<GOAL>

## Kriteria selesai
<DONE_CRITERIA>

## Batasan
- Dilarang menyentuh: <RESERVED_ENV>
- Training berat: jalankan di <GPU_HOST>, bukan di VPS
- Maks iterasi: <MAX_ITER>

## Larangan keras
- Dilarang mengedit/menghapus file test
- Dilarang mengubah definisi metrik
- Dilarang `--force`, `reset --hard`, `rm -rf`
```

**GATE 1 — manusia cek struktur + GOAL.md, ketik `LANJUT`.**

---

## FASE 2 — KONFIGURASI MODEL

| peran | model | effort | tugas |
|---|---|---|---|
| **eksekutor** | `sonnet` | low | tulis kode, jalankan, baca error, perbaiki |
| **perencana** | `opus` | medium | diagnosis, brainstorming, opsi arsitektur |
| **peneliti** | manusia + claude.ai Research | — | pertanyaan yang benar-benar mahal |

Effort diatur lewat thinking budget:

```bash
# eksekutor — low
export MAX_THINKING_TOKENS=2000

# perencana — medium
export MAX_THINKING_TOKENS=10000
```

> **VERIFIKASI DULU**: nama env var thinking budget bisa berubah antar versi Claude Code.
> Cek `claude --help` dan dokumentasi resmi sebelum bergantung padanya.
> Kalau tidak tersedia, turunkan effort lewat instruksi prompt: eksekutor diminta
> "jangan analisis panjang, langsung eksekusi dan ukur".

---

## FASE 3 — TABEL ESKALASI

Naik tingkat berdasarkan **jenis error**, bukan cuma hitungan gagal.

| lvl | pemicu | siapa | tools | output |
|---|---|---|---|---|
| 1 | gagal ke-1..2, syntax/import/typo | sonnet low | Bash, Edit, Read | commit atau naik |
| 2 | gagal ke-3 | sonnet low | + Read docs lokal, `--help`, `pip show` | commit atau naik |
| 3 | gagal ke-4 | sonnet low | + WebSearch (maks 3 query) | commit atau naik |
| 4 | gagal ke-5, ATAU API deprecated / signature berubah | **opus medium** | WebSearch + WebFetch penuh, maks 10 sumber | `BRIEF.md` |
| 5 | gagal ke-6, ATAU pertanyaan arsitektur/metode | **opus medium** | + Task (maks 5 subagent, maks 8 fetch masing-masing) | `RESEARCH.md` internal |
| 6 | gagal ke-7, ATAU lvl 5 tidak menghasilkan jalan | **STOP** | — | `BLOCKED.md` + notif manusia |

**Lompat langsung**:
- error menyebut API/versi library berubah → langsung lvl 4 (knowledge cutoff basi)
- pertanyaan "arsitektur mana yang benar" → langsung lvl 5
- OOM / hardware / kuota → langsung lvl 6, ini masalah infrastruktur bukan riset

**Reset hitungan** hanya kalau ada commit hijau baru.

---

## FASE 4 — TEMPLATE HANDOFF

Format wajib. Agent dilarang mengarang bagian.

### `handoff/STUCK.md` — ditulis eksekutor

```markdown
# STUCK
run: <RUN_ID>   iterasi: <n>   level: <lvl>   waktu: <ISO8601>

## Perintah yang dijalankan
```
<perintah persis>
```

## Error mentah
```
<paste utuh, jangan diringkas, maks 200 baris>
```

## Hipotesis
1. ...
2. ...

## Sudah dicoba dan gagal
| percobaan | hasil |
|---|---|
| ... | ... |

## Yang belum dicoba dan kenapa
- ...

## File tersentuh
- ...

## Commit terakhir hijau
<sha>
```

### `handoff/BRIEF.md` — ditulis perencana (lvl 4)

```markdown
# BRIEF
menjawab: STUCK.md iterasi <n>   waktu: <ISO8601>

## Diagnosis
<akar masalah, bukan gejala>

## Bukti
| klaim | sumber (URL) | tanggal sumber |
|---|---|---|

## Opsi
### Opsi A — <nama>
- cara: ...
- biaya/risiko: ...
### Opsi B — <nama>
### Opsi C — <nama>

## Konflik antar sumber
<tulis eksplisit. jangan dirata-rata.>

## Yang TIDAK ketemu
<wajib diisi. kalau kosong tulis "tidak ada", tapi pikir dulu.>

## Rekomendasi
<satu opsi + alasan>

## Langkah eksekusi
1. ...
2. ...

## Keyakinan: tinggi / sedang / rendah
```

### `handoff/RESEARCH.md` — lvl 5 internal, atau hasil tempel dari claude.ai

```markdown
# RESEARCH
sumber: <"subagent internal" | "claude.ai Research, <tanggal>">
menjawab: <pertanyaan utama>

## Pertanyaan
## Sub-pertanyaan
## Temuan            (tiap klaim → URL + tanggal)
## Konflik antar sumber
## Yang TIDAK ketemu
## Opsi + trade-off
## Rekomendasi + alasan
## Keyakinan
```

---

## FASE 5 — `BLOCKED.md` (LEVEL 6)

Ini **produk utama mode B**. Isinya prompt siap tempel ke claude.ai dengan Research aktif.

Agent menulis file ini lalu **berhenti total**.

```markdown
# BLOCKED
run: <RUN_ID>   iterasi: <n>   waktu: <ISO8601>
commit terakhir hijau: <sha>

## Ringkasan buntu (3 kalimat)
...

## Kenapa berhenti
<mis. "lvl 5 menghasilkan 2 opsi yang saling bertentangan tanpa bukti pemutus">

## Yang sudah dicoba (ringkas)
| lvl | aksi | hasil |
|---|---|---|

---

## PROMPT SIAP TEMPEL — claude.ai + Research ON

> Salin blok di bawah utuh ke claude.ai. Aktifkan **Research**. Aktifkan **Web search**.

```
Saya butuh riset mendalam untuk memutuskan satu hal teknis. Konteks di bawah.

## Konteks proyek
Tujuan: <GOAL>
Kriteria selesai: <DONE_CRITERIA>
Stack: <bahasa, framework, versi persis>
Lingkungan: <VPS spek / GPU host / OS>
Batasan keras: <mis. tanpa GPU di VPS, budget nol, harus lokal>

## Masalah
<deskripsi buntu, teknis, spesifik>

## Error mentah
<paste blok error>

## Sudah dicoba dan gagal
1. <aksi> → <hasil>
2. ...

## Pertanyaan utama
<satu kalimat, bisa dijawab>

## Sub-pertanyaan
1. ...
2. ...
3. ...
4. ...
5. ...

## Yang saya butuhkan dari jawaban
- Diagnosis akar masalah, bukan gejala
- Minimal 3 opsi dengan trade-off nyata
- Tiap klaim teknis disertai URL sumber + tanggal sumber
- Kalau sumber saling bertentangan, tulis konfliknya eksplisit, jangan dirata-rata
- Bagian "yang tidak ketemu" — apa yang tidak ada jawabannya di sumber publik
- Satu rekomendasi + alasan
- Tingkat keyakinan: tinggi/sedang/rendah

## Format keluaran
Markdown, dengan heading persis ini:
## Pertanyaan
## Temuan
## Konflik antar sumber
## Yang TIDAK ketemu
## Opsi + trade-off
## Rekomendasi + alasan
## Keyakinan

Jangan tulis kode implementasi. Saya butuh keputusan, bukan patch.
```

---

## Cara melanjutkan setelah dapat jawaban

1. Simpan jawaban claude.ai ke `handoff/RESEARCH.md`
2. Arsipkan salinan ke `research/<RUN_ID>-<slug>.md`
3. Catat di ledger: `## <waktu> | HUMAN-RESEARCH | <slug> | <keputusan>`
4. Jalankan:

```bash
cd $PROJECT_ROOT
MAX_THINKING_TOKENS=2000 claude -p "Baca handoff/GOAL.md, handoff/RESEARCH.md, handoff/STUCK.md. \
Terapkan rekomendasi. Reset hitungan gagal ke 0. Patuhi SAFETY RULES." \
  --model sonnet \
  --allowedTools "Read,Write,Edit,Bash"
```
```

---

## FASE 6 — PERINTAH SIAP TEMPEL

Jalankan di dalam `tmux` pada VPS. Laptop boleh ditutup.

```bash
tmux new -s agent-$RUN_ID
cd $PROJECT_ROOT
```

### Eksekutor (lvl 1-3)

```bash
MAX_THINKING_TOKENS=2000 claude -p "$(cat <<'EOF'
Skill: caveman (ultra), ponytail (full), safety-guard, terminal-ops.

Baca handoff/GOAL.md dan STATE.json.
Kerjakan iterasi berikutnya menuju kriteria selesai.

ATURAN:
- Solusi paling malas yang jalan. YAGNI. Jangan tambah dependency tanpa alasan tertulis.
- Jangan analisis panjang. Eksekusi, ukur, laporkan.
- Setiap perubahan berhasil: git commit. Pesan commit gaya caveman-commit.
- Gagal 1-2x: perbaiki sendiri.
- Gagal 3x: baca docs lokal, --help, pip show.
- Gagal 4x: WebSearch maks 3 query.
- Gagal 5x: BERHENTI. Tulis handoff/STUCK.md format lengkap. Jangan lanjut.
- Error menyebut API/signature/versi berubah: langsung tulis STUCK.md, tandai level 4.
- OOM / kuota / hardware: langsung tulis handoff/BLOCKED.md, berhenti.

DILARANG: edit file test, ubah definisi metrik, sentuh <RESERVED_ENV>,
git push, git reset --hard, rm -rf, --force.

Catat tiap aksi ke runs/<RUN_ID>/ledger.md dengan format:
## <ISO8601> | <lvl> | <aksi> | <hasil> | <sha atau ->
EOF
)" \
  --model sonnet \
  --allowedTools "Read,Write,Edit,Bash,WebSearch" \
  --output-format stream-json --verbose \
  | tee -a runs/$RUN_ID/events.jsonl
```

### Perencana lvl 4 — BRIEF

```bash
MAX_THINKING_TOKENS=10000 claude -p "$(cat <<'EOF'
Skill: research-ops, ponytail (untuk menilai opsi mana yang paling sederhana).

Baca handoff/STUCK.md dan handoff/GOAL.md.
Riset penyebab dan solusi. WebFetch sumber PENUH, jangan andalkan snippet.
Maks 10 sumber. Utamakan dokumentasi resmi, changelog, issue tracker.
Perhatikan tanggal sumber — knowledge cutoff bisa basi.

Tulis handoff/BRIEF.md dengan format wajib dari PLAN.

DILARANG KERAS: menyentuh file apa pun di code/. Kamu read-only terhadap kode.
Satu-satunya file yang boleh kamu tulis: handoff/BRIEF.md dan ledger.
EOF
)" \
  --model opus \
  --allowedTools "Read,Write,WebSearch,WebFetch" \
  --output-format stream-json --verbose \
  | tee -a runs/$RUN_ID/events.jsonl
```

### Perencana lvl 5 — RESEARCH internal

```bash
MAX_THINKING_TOKENS=10000 claude -p "$(cat <<'EOF'
Skill: research-ops, literature-review, parallel-execution-optimizer.

Baca handoff/STUCK.md, handoff/BRIEF.md, handoff/GOAL.md.

Pecah jadi 3-5 sub-pertanyaan. Spawn subagent paralel via Task.
BATAS KERAS: maksimal 5 subagent. Maksimal 8 web_fetch per subagent.
Lewat batas = berhenti dan laporkan.

Sumber: WebSearch, WebFetch, arXiv API, Semantic Scholar API (on-demand saja).
Baca sumber PENUH. Konflik antar sumber ditulis eksplisit, jangan dirata-rata.
Bagian "Yang TIDAK ketemu" wajib diisi jujur.

Tulis handoff/RESEARCH.md format wajib.
Arsipkan salinan ke research/<RUN_ID>-<slug>.md.

Kalau hasil riset TIDAK memberi jalan yang jelas:
tulis handoff/BLOCKED.md lengkap dengan PROMPT SIAP TEMPEL, lalu BERHENTI.

DILARANG KERAS: menyentuh file di code/.
EOF
)" \
  --model opus \
  --allowedTools "Read,Write,WebSearch,WebFetch,Task,Bash" \
  --output-format stream-json --verbose \
  | tee -a runs/$RUN_ID/events.jsonl
```

Detach: `Ctrl+B` lalu `D`. Balik: `tmux a -t agent-$RUN_ID`.

---

## FASE 7 — PENCATATAN UNTUK PENELITIAN

Ini eksperimen. Data proses = data penelitian. Jangan sampai hilang.

### `runs/<RUN_ID>/ledger.md` — append-only, wajib tiap aksi

```
## <ISO8601> | lvl <n> | <AKSI> | <HASIL> | <sha>
```

`AKSI` ∈ `EXEC | SEARCH | BRIEF | RESEARCH | HUMAN-RESEARCH | BLOCKED | RESUME`

### `STATE.json`

```json
{
  "run_id": "...",
  "iterasi": 0,
  "level": 1,
  "gagal_beruntun": 0,
  "commit_hijau_terakhir": "",
  "status": "running | blocked | done",
  "menunggu_manusia": false,
  "diperbarui": "<ISO8601>"
}
```

### Yang wajib diarsipkan

- [ ] `events.jsonl` — stream mentah tiap sesi (dari `--output-format stream-json`)
- [ ] Semua versi `STUCK.md` / `BRIEF.md` / `RESEARCH.md` — jangan ditimpa, salin ke `runs/<RUN_ID>/transcripts/<n>-<nama>.md` sebelum ditulis ulang
- [ ] Setiap `git commit` = satu titik eksperimen. Pesan commit sebutkan level dan iterasi
- [ ] Jawaban dari claude.ai Research: simpan **utuh**, termasuk daftar sumbernya
- [ ] Perkiraan token/biaya per level — untuk menjawab "eskalasi bertingkat itu hemat atau tidak"

### Sinkron ke vault (opsional, setelah run selesai)

- Hasil riset yang terbukti berguna → `~/research-vault/literature-notes/`
- Paper yang dikutip → Zotero + Better BibTeX
- **Gate manusia**: tidak ada yang masuk vault tanpa kamu baca dulu

---

## FASE 8 — SKILL YANG DIPAKAI

| skill | dipakai di | fungsi |
|---|---|---|
| `caveman` (ultra) | eksekutor | tekan token output |
| `ponytail` | eksekutor + penilaian opsi | tolak over-engineering |
| `caveman-commit` | tiap commit | pesan commit padat |
| `safety-guard` | eksekutor | cegah operasi destruktif |
| `terminal-ops` | eksekutor | eksekusi berbasis bukti |
| `research-ops` | lvl 4, 5 | riset evidence-first |
| `literature-review` | lvl 5 | screening + sintesis paper |
| `parallel-execution-optimizer` | lvl 5 | atur subagent paralel |
| `strategic-compact` | sesi panjang | compact di batas fase, bukan acak |
| `verification-loop` | sebelum tandai selesai | verifikasi klaim "sudah beres" |

> **Catatan**: skill bernama `superuser` **tidak ada** di daftar terpasang.
> Yang paling dekat dengan maksudnya: `safety-guard` + `terminal-ops`.
> Kalau kamu punya skill itu di tempat lain, tambahkan namanya ke prompt eksekutor.
> Cek dengan `ls ~/.claude/skills`.

---

## SAFETY RULES

Berlaku di semua level. Tidak bisa ditawar oleh agent.

1. **Dilarang** `git push`, `git reset --hard`, `git clean -fdx`, `rm -rf`, flag `--force`
2. **Dilarang** mengedit atau menghapus file test
3. **Dilarang** mengubah definisi metrik atau ambang kriteria selesai
4. **Dilarang** menyentuh `<RESERVED_ENV>` (venv Scrapy dan lingkungan produksi lain)
5. **Dilarang** mengubah konfigurasi SSH, firewall, atau WireGuard
6. **Dilarang** menulis kunci privat, token, atau `.env` ke repo
7. **Dilarang** `--dangerously-skip-permissions` di luar container sekali-pakai
8. Perencana (opus) **read-only** terhadap `code/`. Hanya boleh menulis di `handoff/`, `research/`, `runs/`
9. Training berat **wajib** di `<GPU_HOST>`. VPS low-spec hanya menjalankan otak agent
10. Lewat `MAX_ITER` → berhenti, tulis `BLOCKED.md`, jangan lanjut
11. Lewat `MAX_SUBAGENT` → berhenti dan laporkan
12. Ragu = berhenti dan tanya. Jangan tebak lalu jalan

---

## GATE MANUSIA

| gate | kapan | aksi |
|---|---|---|
| **G0** | setelah RECON | baca ringkasan, ketik `LANJUT` |
| **G1** | setelah scaffold + GOAL.md | cek isi, ketik `LANJUT` |
| **G2** | `BLOCKED.md` muncul | tempel prompt ke claude.ai Research, simpan hasil |
| **G3** | sebelum masuk vault/Zotero | baca hasil riset, setujui |
| **G4** | sebelum tandai `done` | jalankan `verification-loop`, verifikasi kriteria selesai sendiri |

Agent **tidak boleh** melewati gate sendiri.

---

## .gitignore

```gitignore
# rahasia
.env
.env.*
*.pem
*.key
id_rsa*
wg*.conf

# lingkungan
.venv/
venv/
__pycache__/
node_modules/

# artefak berat
*.ckpt
*.pt
*.pth
*.safetensors
data/raw/
checkpoints/

# log mentah — besar, tapi ini data penelitian
# pilih salah satu:
#   commit (kalau kecil, demi reproducibility)  → komentari baris di bawah
#   atau arsip terpisah                          → biarkan aktif
runs/*/events.jsonl
runs/*/transcripts/

# JANGAN abaikan ini — inti penelitian
!runs/*/ledger.md
!research/
!handoff/
```

---

## CATATAN JUJUR

- **Level 6 wajib ada.** Tanpa itu loop tidak pernah berhenti dan token habis semalaman.
- **Eskalasi bertingkat belum tentu lebih hemat.** Kadang langsung opus lebih murah daripada sonnet gagal 5x. Ledger-mu yang akan menjawab ini — itu justru temuan penelitian yang berguna.
- **Research claude.ai jarang dibutuhkan.** Perkiraan 1-2x per proyek, biasanya soal arsitektur di awal. Kalau tiap hari kena lvl 6, masalahnya di `GOAL.md` yang kabur, bukan di loop.
- **Agent akan curang kalau bisa.** Menghapus test agar hijau adalah kejadian nyata, bukan hipotesis. Aturan 2 dan 3 bukan formalitas.
- **`MAX_THINKING_TOKENS` perlu diverifikasi** terhadap versi Claude Code yang kamu pakai sekarang.