# Protokol reader study: kegunaan klinis penjelasan `fusion_late`

**Status: belum dijalankan. Nol data manusia dikumpulkan.** Dokumen ini ditulis
19 Agustus 2026 supaya studinya bisa dimulai tanpa mendesain ulang, dan supaya
Limitations manuskrip bisa menyebut absennya evaluasi klinisi sebagai kekosongan
yang sudah dipetakan, bukan yang belum dipikirkan.

---

## 1. Kenapa ini dibutuhkan

Klaim utama Track 1 setelah §6.4 laporan berbunyi: `fusion_late` ekuivalen dengan
`radiomics_only` dalam AUC, dan satu-satunya arm yang menyediakan penjelasan spasial
serta penjelasan fitur sekaligus. Kalimat kedua itu **pernyataan tentang kapabilitas,
bukan tentang manfaat.** Ia membuktikan kedua penjelasan itu ada; ia tidak membuktikan
ada orang yang bisa memakainya.

Semua metrik XAI yang dilaporkan sejauh ini — dice, IoU, pointing accuracy, energy
pointing — mengukur kecocokan peta dengan mask radiolog. Tidak satu pun menanyakan
apakah peta itu membantu seorang dokter memutuskan sesuatu. Jarak antara "peta jatuh
di dalam nodul" dan "peta mengubah keputusan menjadi lebih baik" tidak bisa dijembatani
dengan menambah metrik otomatis.

Studi ini menutup jarak itu. Ia **tidak** dimaksudkan untuk memperkuat klaim AUC, yang
sudah punya uji sendiri.

## 2. Pertanyaan penelitian

- **RQ1.** Apakah penjelasan gabungan (peta spasial + atribusi fitur) dinilai lebih
  berguna daripada peta spasial saja untuk menilai malignansi nodul?
- **RQ2.** Apakah pembaca sepakat satu sama lain soal kegunaan itu, atau penilaiannya
  murni selera pribadi?
- **RQ3.** Apakah kepercayaan pembaca pada prediksi model berubah ketika penjelasannya
  ditambahkan — dan apakah perubahan itu terkalibrasi, yaitu naik pada prediksi benar
  dan turun pada prediksi salah?

RQ3 sengaja dipisah karena penjelasan yang menaikkan kepercayaan secara seragam,
termasuk pada prediksi yang salah, **berbahaya**, bukan berguna. Studi yang hanya
mengukur "apakah pembaca suka" tidak bisa membedakan keduanya.

## 3. Pembaca

| Butir | Ketetapan |
|---|---|
| Jumlah | 3 minimum, 5 diutamakan |
| Kualifikasi | Radiolog atau residen radiologi tahun ≥3, dengan pengalaman membaca CT toraks |
| Stratifikasi | Sedikitnya satu pembaca senior (>5 tahun pasca-spesialisasi) dan satu junior, supaya efek pengalaman terlihat alih-alih tercampur |
| Pelatihan | Satu sesi kalibrasi 15 menit memakai 2 nodul di luar himpunan studi, untuk menjelaskan arti peta Layer-CAM dan plot SHAP. Data sesi ini dibuang |
| Kompensasi dan etik | Ditentukan institusi. Studi ini tidak mengumpulkan data pasien baru; seluruh citranya LIDC-IDRI yang sudah publik dan dianonimkan |

Tiga pembaca adalah lantai, bukan target. Dengan tiga pembaca, inter-rater agreement
bisa dilaporkan tapi interval kepercayaannya lebar, dan itu wajib dinyatakan di hasil.

## 4. Kasus

**Sumber kasus: `artifacts/xai/fixed_display_samples.json`. Tidak dipilih ulang.**

Berkas itu memuat enam nodul yang dibekukan sebelum analisis XAI mana pun dijalankan
(`S1` LIDC-IDRI-0732/1, `S2` -0194/1, `S3` -0075/1, `S4` -0164/10 berlabel malignan;
`S5` -0491/1, `S6` -0469/0 berlabel benigna), lengkap dengan `patch_path`, `mask_path`,
dan probabilitas per backbone saat diseleksi. GOAL2 melarang mengubahnya, dan larangan
itu berlaku penuh di sini: memilih ulang kasus untuk reader study akan menghapus
satu-satunya jaminan bahwa himpunan tampilannya tidak dipilih setelah melihat hasilnya.

Enam kasus terlalu sedikit untuk daya statistik. Perluasan yang **sah**:

1. Enam nodul tetap sebagai inti wajib, dinilai seluruh pembaca. Ini jangkar
   perbandingan dengan angka §6.3.4 yang sudah terbit.
2. Tambahan diambil dari himpunan 60-nodul fold 0 yang sudah dipakai `stage_05_xai`
   dan `stage_08b_run02_xai` (`df[(fold==0) & (label!=-1)].sample(n=60, random_state=42)`),
   **memakai seed yang sama**, sehingga himpunannya dapat direproduksi dan tidak
   dikurasi. Target 30 kasus dari 60 itu, diambil dari indeks terkecil ke atas — bukan
   dipilih berdasarkan tampilan petanya.
3. Nodul dengan `n_disagree` positif, yaitu yang `cnn_only` dan `fusion_late`-nya
   memprediksi kelas berbeda, **wajib** ikut, karena hanya di sanalah kedua peta
   benar-benar berbeda. Jumlahnya 5, 6, dan 2 dari 59 tergantung backbone.

Butir 3 penting. Kalau kasusnya diambil acak saja, sebagian besar akan berupa nodul
yang kedua arm-nya sepakat, dan di sana petanya identik bit-per-bit. Studi seperti itu
akan membandingkan dua gambar yang sama lalu menyimpulkan tidak ada beda.

## 5. Kondisi

Setiap kasus disajikan dalam tiga kondisi:

| Kode | Isi yang ditampilkan |
|---|---|
| **A** | Patch CT + prediksi probabilitas model. Tanpa penjelasan |
| **B** | A + peta Layer-CAM (penjelasan spasial saja — setara kemampuan `cnn_only`) |
| **C** | B + plot SHAP cabang radiomik (penjelasan gabungan — hanya `fusion_late`) |

Kondisi setara `radiomics_only` (SHAP tanpa peta) **tidak** disertakan, dan alasannya
harus dicatat: arm itu tidak pernah melihat citra, sehingga menampilkan patch CT
bersamanya akan menyiratkan model memakai informasi yang tidak dimilikinya.

Mask radiolog **tidak** ditampilkan ke pembaca. Ia jawaban, bukan masukan.

## 6. Desain

- **Within-reader, kasus disilang kondisi.** Setiap pembaca melihat setiap kasus tepat
  satu kali, dalam satu kondisi, dengan penugasan kondisi diputar antar-pembaca (desain
  Latin square) sehingga setiap kasus terlihat pada ketiga kondisi di seluruh panel
  tanpa satu pembaca pun melihat kasus yang sama dua kali.
- **Urutan diacak** per pembaca, seed dicatat.
- **Blinded**: pembaca tidak diberi tahu label ground truth, arm mana yang menghasilkan
  prediksi, backbone mana yang dipakai, maupun apakah prediksi model benar.
- **Tanpa batas waktu**, tapi waktu per kasus dicatat. Penjelasan yang berguna tapi
  menuntut tiga menit per nodul punya biaya yang harus terlihat.
- **Tidak ada umpan balik** antar-kasus. Pembaca tidak pernah tahu skornya.

Melihat kasus yang sama dua kali akan membuat penilaian kondisi kedua terkontaminasi
ingatan kondisi pertama. Itu sebabnya desainnya menyilangkan, bukan mengulang.

## 7. Instrumen

Per kasus, pembaca mengisi:

1. **Penilaian malignansi sendiri**, skala 1–5 seperti konvensi LIDC. Diambil *sebelum*
   penjelasan diperlihatkan pada kondisi B dan C, lalu boleh direvisi sesudahnya —
   kedua nilainya dicatat.
2. **Kepercayaan pada prediksi model**, 1–7 (1 = sama sekali tidak percaya,
   7 = sepenuhnya percaya).
3. **Kegunaan penjelasan**, 1–7, hanya pada kondisi B dan C. Pertanyaannya dirumuskan
   spesifik: *"Seberapa jauh penjelasan ini membantu Anda menilai apakah prediksi model
   layak dipercaya untuk kasus ini?"* — bukan "seberapa Anda menyukai penjelasan ini".
4. **Apakah penjelasan menunjuk bukti yang relevan secara klinis**, ya / tidak / tidak yakin.
5. **Komentar bebas**, opsional.

Skala 7 titik dipakai untuk butir 2 dan 3 supaya ada ruang bergerak; skala 5 titik pada
butir 1 dipertahankan agar sebanding dengan rating LIDC asli.

## 8. Analisis

| Pertanyaan | Uji |
|---|---|
| RQ1: C lebih berguna dari B? | Model campuran linear dengan efek acak untuk pembaca dan kasus, efek tetap kondisi. Alternatif non-parametrik: Wilcoxon signed-rank berpasangan per kasus |
| RQ2: pembaca sepakat? | Krippendorff's alpha untuk data ordinal, dilaporkan dengan interval bootstrap. ICC(2,k) sebagai pelengkap |
| RQ3: kepercayaan terkalibrasi? | Selisih kepercayaan (C dikurangi A) dipecah menurut benar atau salahnya prediksi model. Yang dicari **interaksi**, bukan efek utama |
| Biaya waktu | Waktu per kasus per kondisi, dilaporkan sebagai median dan IQR |

**Ditetapkan sebelum data masuk:** hipotesis utamanya RQ1; RQ2 dan RQ3 dinyatakan
eksploratoris. Dengan 3–5 pembaca, studi ini tidak bertenaga untuk mendeteksi efek
kecil, dan itu dilaporkan sebagai batasan alih-alih disamarkan lewat p-hacking pada
subkelompok.

Hasil **nol** — C tidak dinilai lebih berguna dari B — dilaporkan apa adanya. Itu hasil
yang informatif: ia berarti atribusi fitur tabular tidak menambah nilai yang bisa
dirasakan pembaca, dan klaim kapabilitas Track 1 harus dibaca lebih hati-hati.

## 9. Bahan yang perlu disiapkan

| Bahan | Status | Sumber |
|---|---|---|
| Patch CT per kasus | Ada | `patch_path` di `fixed_display_samples.json`; `artifacts/patches/` |
| Peta Layer-CAM per kasus | Ada untuk 6 nodul tetap | `artifacts/results/run02/fig14_spatial_and_feature.png` panel atas, dibangkitkan `src/stage_08d_run02_fig14.py` |
| Plot SHAP | Ada, satu untuk seluruh backbone | `artifacts/results/run02/shap_beeswarm_{backbone}.png` — ketiganya identik menurut konstruksi (§6.3.5) |
| Panel per kasus terpisah | **Belum** | Butuh skrip kecil yang memotong figur gabungan jadi satu panel per kasus per kondisi |
| Lembar skor | **Belum** | Bisa berupa form digital; butir §7 sudah final |
| Peta untuk 30 kasus tambahan | **Belum** | Butuh menjalankan CAM pada himpunan 60-nodul, bukan hanya 6 |

Tiga baris terakhir adalah pekerjaan yang tersisa sebelum studi bisa dimulai.
Bebannya kecil dibanding fase faithfulness (`handoff/GOAL3.md`), dan tidak bergantung
padanya.

## 10. Yang studi ini tidak bisa jawab

- Apakah model layak dipakai secara klinis. Enam sampai 36 kasus dengan 3–5 pembaca
  jauh dari cukup untuk itu.
- Apakah `fusion_late` lebih baik dari radiolog. Perbandingan itu tidak dirancang di sini.
- Apakah penjelasannya *faithful*, yaitu benar-benar mencerminkan komputasi model. Itu
  pertanyaan berbeda, dijawab metrik perturbasi di `handoff/GOAL3.md`, dan sebuah
  penjelasan bisa saja dinilai berguna oleh pembaca sekaligus tidak faithful.

Butir terakhir tidak boleh dilupakan saat melaporkan hasil: penjelasan yang meyakinkan
tapi tidak faithful adalah kegagalan yang menyamar sebagai keberhasilan.
