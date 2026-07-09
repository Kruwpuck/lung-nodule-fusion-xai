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
