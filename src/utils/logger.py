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

    If the file already exists with a different header, only keys present in
    both the existing header and `row` are written (extra keys in `row` are
    dropped) to avoid corrupting the file as the schema grows over time.
    """
    import csv as _csv
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    if new:
        fieldnames = list(row.keys())
    else:
        with open(path, "r", newline="") as f:
            existing_header = next(_csv.reader(f))
        fieldnames = existing_header
        row = {k: row.get(k) for k in fieldnames}
    with open(path, "a", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        if new:
            writer.writeheader()
        writer.writerow(row)
