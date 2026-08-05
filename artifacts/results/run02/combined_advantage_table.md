# Tabel keunggulan gabungan (2026-08-04-run02, commit 5220afb)

| criterion                                              | cnn_only                               | radiomics_only                             | fusion_late                      |
|:-------------------------------------------------------|:---------------------------------------|:-------------------------------------------|:---------------------------------|
| AUC gabungan 5 fold (rerata 3 backbone)                | 0.8944 (0.8907-0.8965)                 | 0.9336 (0.9336-0.9336)                     | 0.9327 (0.9300-0.9363)           |
| AUC dengan checkpoint tanpa seleksi (T-0)              | 0.8806 (0.8725-0.8888)                 | 0.9336 (0.9336-0.9336) (tidak terpengaruh) | 0.9290 (0.9241-0.9360)           |
| DeLong vs fusion_late (p, 3 backbone)                  | 0.0000 (0.0000-0.0000); signifikan 3/3 | 0.5417 (0.4555-0.6793); signifikan 0/3     | -                                |
| Peta salience spasial (Grad-CAM/Layer-CAM)             | ada                                    | mustahil secara struktural                 | ada (diwarisi dari cabang citra) |
| Pointing accuracy (60 nodul fold 0, rerata 3 backbone) | 0.6667 (0.5763-0.7288)                 | tidak terdefinisi                          | 0.6384 (0.5593-0.6780)           |
| SHAP fitur radiomik                                    | tidak ada                              | ada                                        | ada                              |
| Penjelasan spasial DAN fitur sekaligus                 | tidak                                  | tidak                                      | ya                               |
