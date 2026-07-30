"""CSV epoch logger — flush per row for hang safety."""
import csv
import os


class CSVLogger:
    def __init__(self, path: str, fieldnames: list) -> None:
        new = not os.path.exists(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.f = open(path, "a", newline="")
        self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
        if new:
            self.writer.writeheader()

    def log(self, row: dict) -> None:
        self.writer.writerow(row)
        self.f.flush()

    def close(self) -> None:
        self.f.close()


def append_row(path: str, row: dict) -> None:
    """Append one row to a run-level CSV, creating header from `row` keys if new.

    If the file already exists and `row` introduces keys not in the existing
    header (e.g. a new experiment records fields older runs never had), the
    header — and every existing row — is extended with those columns
    (backfilled empty) so the new data is never silently dropped. This keeps
    `runs.csv` additive: old rows and their columns are untouched, new columns
    just start out blank for them.
    """
    import csv as _csv
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    if new:
        with open(path, "a", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    with open(path, "r", newline="") as f:
        reader = _csv.DictReader(f)
        existing_header = reader.fieldnames or []
        existing_rows = list(reader)

    extra_keys = [k for k in row.keys() if k not in existing_header]
    if extra_keys:
        fieldnames = existing_header + extra_keys
        with open(path, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in existing_rows:
                writer.writerow(r)
            writer.writerow({k: row.get(k, "") for k in fieldnames})
        return

    with open(path, "a", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=existing_header)
        writer.writerow({k: row.get(k) for k in existing_header})
