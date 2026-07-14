from src.data_loading.lidc_loader import load_and_split

df = load_and_split(
    lidc_path='data/raw/LIDC-IDRI',
    interim_path='data/interim',
    processed_path='data/processed',
    n_folds=5,
    seed=42,
)
print(f"Nodules: {len(df)} | Malignant: {(df.label==1).sum()} | Benign: {(df.label==0).sum()}")
