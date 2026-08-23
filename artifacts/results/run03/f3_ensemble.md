# F-3 — ensembling tiga backbone di dalam fold luar yang sama

`run_id` 2026-08-22-run03 · sumber `run03/ensemble.csv` (24 baris) ·
kode `src/stage_10c_ensemble.py`

Ensembling antar-**fold** tidak tersedia: tiap fold menahan pasien yang berbeda,
jadi tidak ada satu kasus pun yang punya prediksi luar-fold dari dua model fold
sekaligus. Yang tersedia sumbu satunya — di dalam satu fold luar, ketiga backbone
menilai pasien tertahan yang sama, jadi probabilitasnya boleh dirata tanpa satu
kasus pun dinilai model yang pernah melihatnya saat latih.

---

## 1. Prasyarat: urutan kasus, dibuktikan bukan diasumsikan

Merata-rata probabilitas secara posisional hanya sah kalau ketiga array memuat
nodul yang sama dalam urutan yang sama. Secara konstruksi memang begitu — frame
`merged` yang sama, filter fold yang sama, `reset_index(drop=True)`, dan
DataLoader `shuffle=False`. Argumen semacam itu tetap benar sampai seseorang
mengubah salah satu dari empat hal tersebut, dan ketidaksejajaran **tidak akan
melempar error**: ia akan diam-diam merata probabilitas CNN pasien A ke label
pasien B lalu menurunkan setiap AUC beberapa poin.

Karena itu urutan di-assert terhadap `patient_id`, `nodule_idx` dan `fold`, dan
buktinya dicetak sebelum satu probabilitas pun dirata:

| fold | n | sha1(urutan), ketiga backbone | kasus pertama | kasus terakhir | positif |
|---|---|---|---|---|---|
| 0 | 292 | `c8d2aaff0d` | LIDC-IDRI-0136#2 | LIDC-IDRI-0701#3 | 93/292 |
| 1 | 254 | `41e8a75b53` | LIDC-IDRI-0101#0 | LIDC-IDRI-0721#3 | 81/254 |
| 2 | 266 | `ebfd916ba8` | LIDC-IDRI-0069#1 | LIDC-IDRI-0704#1 | 92/266 |
| 3 | 303 | `a46a0d4a56` | LIDC-IDRI-0003#0 | LIDC-IDRI-0127#0 | 88/303 |
| 4 | 251 | `88e26dfb08` | LIDC-IDRI-0078#0 | LIDC-IDRI-0705#0 | 84/251 |

Satu digest per fold, identik pada ketiga backbone, total 1366 nodul. Identik
juga pada kedua rezim checkpoint — yang memang harus, karena checkpoint mengubah
probabilitas, bukan urutan kasus.

`--self-check` membuktikan penjaganya benar-benar menyala: dua kasus di dalam
satu fold ditukar posisinya (himpunan sama, urutan salah) dan assert-nya
melempar. Penjaga yang tidak pernah menyala cuma hiasan.

---

## 2. Dua rezim checkpoint, dua arm

- `selected_published` — probabilitas run02, checkpoint di balik tabel Track 1
  yang terbit, epoch-nya dipilih pada fold luar yang sama yang dipakai menilai.
- `honest_nested_cv` — probabilitas run03, sel 100% unfreeze, epoch dipilih pada
  inner split. Batasnya tetap `f2_sensitivity.md` §6: ini **batas bawah** cabang
  CNN jujur, bukan estimasinya.

Dua arm diensemble: `cnn_only`, cabang yang mau diperbaiki, dan `fusion_late`,
arm yang dilaporkan naskah. Arm fusi lain melatih ulang di atas cabang CNN dan
tidak punya probabilitas tersimpan untuk dirata.

---

## 3. AUC ensemble per fold

| rezim | arm | fold0 | fold1 | fold2 | fold3 | fold4 | pooled |
|---|---|---|---|---|---|---|---|
| terbit | cnn_only | 0,9439 | 0,9414 | 0,9014 | 0,9042 | 0,9171 | 0,9140 |
| terbit | fusion_late | 0,9499 | 0,9608 | 0,9225 | 0,9304 | 0,9426 | 0,9388 |
| jujur | cnn_only | 0,9111 | 0,9072 | 0,8498 | 0,8199 | 0,8635 | 0,8693 |
| jujur | fusion_late | 0,9408 | 0,9600 | 0,9003 | 0,9101 | 0,9242 | 0,9269 |

Pooled selalu di bawah rerata-fold (`cnn_only` terbit: rerata-fold 0,9216 lawan
pooled 0,9140). Itu efek penggabungan, bukan temuan: menyatukan lima fold
menambahkan ragam kalibrasi antar-fold ke satu kurva ROC. Angka pooled dan angka
rerata-fold tidak boleh dibandingkan lintas tabel.

---

## 4. Ensemble lawan backbone tunggal terbaik, pooled

| rezim | arm | ensemble | convnext | dense201 | dense121 | terbaik | Δ | p DeLong |
|---|---|---|---|---|---|---|---|---|
| terbit | cnn_only | 0,9140 | 0,8907 | 0,8959 | 0,8965 | densenet121 | **+0,0175** | **0,0016** |
| terbit | fusion_late | 0,9388 | 0,9300 | 0,9363 | 0,9319 | densenet201 | +0,0025 | 0,3552 |
| jujur | cnn_only | 0,8693 | 0,8467 | 0,8360 | 0,8620 | densenet121 | +0,0073 | 0,1922 |
| jujur | fusion_late | 0,9269 | 0,9195 | 0,9171 | 0,9279 | densenet121 | −0,0010 | 0,6821 |

**Ensembling menaikkan cabang CNN mentah, dan hanya itu.** Di rezim terbit
kenaikannya +0,0175 AUC dengan p=0,0016 — di dalam rentang +1..+2 poin yang
dipra-registrasi. Di rezim jujur kenaikannya menyusut jadi +0,0073 dan tidak lagi
signifikan (p=0,19).

Pada `fusion_late` kenaikan itu hilang: +0,0025 (p=0,36) di rezim terbit, dan
−0,0010 (p=0,68) di rezim jujur. Ensemble tiga backbone tidak lebih baik dari satu
backbone begitu radiomik ikut difusikan.

---

## 5. Ensemble lawan `radiomics_only`, pooled

Pembanding radiomik: nilai terkuat dari tiga (`convnext_tiny`, AUC 0,9336).
Ketiganya berhimpit — sebarannya berasal dari `mutual_info_classif` yang jalan
tanpa `random_state`, bukan dari cabang CNN.

| rezim | arm | AUC ensemble | AUC radiomics | Δ | p DeLong | putusan |
|---|---|---|---|---|---|---|
| terbit | cnn_only | 0,9140 | 0,9336 | −0,0196 | 0,0129 | radiomics menang |
| terbit | fusion_late | 0,9388 | 0,9336 | +0,0051 | 0,2101 | tidak signifikan |
| jujur | cnn_only | 0,8693 | 0,9336 | −0,0643 | 4,0e−12 | radiomics menang |
| jujur | fusion_late | 0,9269 | 0,9336 | −0,0067 | 0,0769 | tidak signifikan |

**Nol ensemble mengalahkan `radiomics_only` di sel mana pun.** Yang paling dekat
`fusion_late` ensemble di rezim terbit: +0,0051 AUC, p=0,2101. Aturan keputusan
pra-registrasi §5 menuntut DeLong p<0,05 untuk menyatakan fusi menang, tanpa
pelonggaran. p=0,21 bukan kemenangan.

---

## 6. Putusan gate F-3

- Urutan kasus terverifikasi dan buktinya tercetak — §1. **Lolos.**
- AUC ensemble per fold tercetak — §3. **Lolos.**
- DeLong lawan backbone tunggal terbaik dan lawan `radiomics_only` tercetak — §4,
  §5. **Lolos.**

**Pencarian performa ditutup.** Pra-registrasi §5: "kalau setelah F-1 dan F-3
fusi tetap tidak menang, pencarian ditutup permanen." F-1 negatif — nol dari
sembilan sel kandidat mengalahkan kontrol. F-3 tidak menghasilkan satu pun
kemenangan signifikan atas radiomics. Dua mekanisme terakhir yang belum dicoba
sudah dicoba, dan keduanya tidak membalikkan kesimpulan.

---

## 7. Konsisten dengan disosiasi F-2

F-2 menemukan `fusion_late` mewarisi persis setengah pergerakan sisi CNN karena
`average_fusion(cnn, rad, weight_cnn=0.5)` menahan sisi radiomiknya tetap. F-3
memperlihatkan aturan yang sama bekerja ke arah sebaliknya: kenaikan +0,0175 pada
cabang CNN tidak sampai ke `fusion_late` sebagai kenaikan yang berarti.

Sebabnya tetap bentuk fusinya. `fusion_late` ensemble sama dengan
`average_fusion` antara ensemble CNN dan ensemble radiomik; ensemble CNN-nya
0,9140 masih di bawah radiomics 0,9336, jadi paruh yang lebih lemah tetap paruh
yang lebih lemah dan langit-langit arm itu tidak bergerak. Menaikkan cabang CNN
tanpa melewati radiomics tidak menaikkan fusi rerata-probabilitas.

---

## 8. Yang tidak boleh disimpulkan dari sini

- **Ini bukan pernyataan bahwa ensembling tidak berguna.** Ia berguna, dan
  terukur: +0,0175 AUC pada `cnn_only` di rezim terbit, signifikan. Yang tidak
  terjadi adalah kenaikan itu menembus ke arm fusi atau melewati radiomics.
- **Selisih dua rezim pada besar kenaikan ensemble (+0,0175 lawan +0,0073) tidak
  ditafsirkan di sini.** Checkpoint run03 berbeda dari resep terbit dalam empat
  hal sekaligus (`f2_sensitivity.md` §6), jadi mengaitkan penyusutan itu ke satu
  sebab akan melampaui yang datanya bisa dukung.
- Tiga backbone, satu dataset, satu ukuran masukan. Bukan pernyataan umum tentang
  ensembling.
- Rerata tak berbobot. Nol pembobotan, nol stacking, nol pemilihan subset
  backbone — semuanya akan memerlukan seleksi di atas fold penilaian, persis
  kebocoran yang run03 ada untuk mengukurnya.
