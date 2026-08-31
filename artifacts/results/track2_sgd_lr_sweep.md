# Rev1 tugas 7 — sweep learning rate khusus SGD

`run_id` 2026-08-30-rev1task7 · kode `src/stage_03c_sweep.py` ·
sumber `artifacts/logs/runs.csv` (60 baris ber-sufiks `_lr`) ·
ringkasan `artifacts/results/track2_sgd_lr_sweep.csv`

Kodenya sudah ada sejak Rev1 dan tidak pernah dieksekusi. Pertanyaan yang ia dirancang
untuk menjawab: defisit SGD sebesar 0,146 AUC yang dilaporkan naskah Track 2 itu properti
SGD, atau artefak karena SGD dijalankan pada learning rate milik Adam.

4 backbone Track 2 × 3 LR × 5 fold, weight decay tetap di default 1e-4. Nol timpaan: run
LR-override mendapat sufiks `_lr{lr:g}` pada `run_id` dan subdirektori checkpoint sendiri,
jadi 215 run lama tetap byte-identik.

---

## 1. AUC rerata lintas 5 fold

| model | adam 1e-4 | sgd 1e-4 | sgd 1e-3 | sgd 1e-2 | sgd 1e-1 |
|---|---|---|---|---|---|
| efficientnet_b0 | 0,8755 | 0,7294 | 0,8779 | **0,9137** | 0,8927 |
| mobilenetv2 | 0,8511 | 0,7560 | 0,9000 | **0,9006** | 0,7592 |
| resnet50 | 0,8989 | 0,7721 | 0,8987 | **0,9017** | 0,8363 |
| vgg16 | 0,9118 | 0,8687 | 0,8978 | **0,9107** | 0,5000 |

Kolom `sgd 1e-4` adalah kolom yang dipakai naskah. Kolom `sgd 1e-2` adalah SGD yang
learning rate-nya wajar.

## 2. Selisih terhadap Adam

| model | pada LR bersama (1e-4) | pada LR SGD terbaik (1e-2) |
|---|---|---|
| efficientnet_b0 | −0,1461 | **+0,0382** |
| mobilenetv2 | −0,0951 | **+0,0495** |
| resnet50 | −0,1268 | **+0,0028** |
| vgg16 | −0,0431 | **−0,0011** |

**Defisitnya hilang seluruhnya, dan pada dua backbone ia berbalik arah.** Pada LR terbaik,
SGD tidak pernah tertinggal lebih dari 0,0011 — di bawah simpangan baku antar fold model
mana pun di tabel ini (0,0108 sampai 0,0401).

## 3. Optimum ada, dan 1e-1 sudah melewatinya

Ini bukan "makin besar makin baik". Pada 1e-1 pelatihannya divergen:

| model | AUC 1e-1 | sd antar fold | fold dengan AUC ≤ 0,55 |
|---|---|---|---|
| vgg16 | 0,5000 | **0,0000** | 5 dari 5 |
| mobilenetv2 | 0,7592 | 0,1642 | 0 |
| resnet50 | 0,8363 | 0,1375 | 0 |
| efficientnet_b0 | 0,8927 | 0,0401 | 0 |

vgg16 tepat 0,5000 pada kelima fold dengan sd nol — itu bukan model yang buruk, itu model
yang tidak belajar sama sekali. mobilenetv2 dan resnet50 tidak divergen tapi simpangan
antar fold-nya melonjak tiga sampai empat kali lipat, yang merupakan tanda ketidakstabilan
yang sama pada tahap lebih awal.

Jadi kurvanya bermodus tunggal dengan puncak di sekitar 1e-2, dan sweep terbit mencuplik
ujung paling kiri kurva itu — untuk SGD saja, karena 1e-4 memang tepat untuk Adam.

---

## 4. Konsekuensi untuk naskah Track 2

Kalimat di `paper/track2/main.tex:293` — *"An SGD-specific learning-rate sweep is
implemented but not yet executed here, and is reported as planned follow-up rather than as
a result"* — sekarang punya hasilnya, dan hasilnya mendukung hipotesis yang kalimat itu
ajukan.

Yang **tidak** berubah: aritmetika ANOVA-nya. η² = 0,4533 untuk optimizer lawan 0,000189
untuk weight decay tetap benar untuk grid yang diukur. Sweep ini nol menyentuh grid itu.

Yang berubah adalah **bacaannya**. Rasio 2400 kali di baris 239 bukan bukti bahwa pilihan
optimizer penting; ia bukti bahwa **learning rate** penting, pada grid yang menyandingkan
optimizer dengan kelayakan learning rate-nya. Ketiga optimizer berbagi satu learning rate,
dan learning rate itu milik Adam. Begitu SGD diberi LR-nya sendiri, faktor optimizer
sebagian besar runtuh.

Kualifikasi itu perlu ditulis **di tempat klaimnya muncul** — baris 239 dan §Discussion —
bukan hanya di Limitations, mengikuti disiplin yang sudah dipakai F-2 pada klaim
ekuivalensi. Batasan 9 di `sec:limits` ("hyperparameter identik lintas arsitektur") juga
sekarang punya bukti kuantitatif, bukan sekadar dugaan.

---

## 5. Yang tidak boleh disimpulkan dari sini

- **Bukan pernyataan bahwa SGD lebih baik dari Adam.** Pada LR terbaiknya SGD setara: tiga
  selisih di dalam ±0,05 dan satu di dalam ±0,002. Yang ditunjukkan kesetaraan setelah
  penalaan, bukan keunggulan.
- **Adam dan AdamW tidak ikut disweep LR-nya.** Grid ini hanya menala SGD, karena hanya
  SGD yang hipotesisnya dipra-registrasi. Menala ketiganya bisa menggeser urutannya lagi,
  dan itu tidak dijalankan.
- **Tiga nilai LR, bukan pencarian.** Optimum sesungguhnya bisa berada di antara 1e-2 dan
  1e-1; yang dibuktikan hanya bahwa 1e-4 jauh di bawahnya dan 1e-1 sudah melewatinya.
- **Nol uji signifikansi berpasangan di sini.** Angka yang disajikan rerata lintas 5 fold
  dengan simpangan bakunya. Selisih pada LR terbaik lebih kecil daripada sebaran antar
  fold, yang justru alasan mengapa selisih itu dibaca sebagai kesetaraan dan bukan sebagai
  peringkat.
