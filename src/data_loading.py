from pathlib import Path
import pandas as pd

def load_table(file_stem: str, data_dir: str | Path = "../data/raw") -> pd.DataFrame:
    data_dir = Path(data_dir)
    possible_files = [
        data_dir / f"{file_stem}.xlsx",
        data_dir / f"{file_stem}.xls",
        data_dir / f"{file_stem}.csv",
    ]

    for path in possible_files:
        if path.exists():
            print(f"Loading: {path}")
            if path.suffix.lower() in [".xlsx", ".xls"]:
                return pd.read_excel(path)
            return pd.read_csv(path)

    raise FileNotFoundError(
        f"Could not find {file_stem}.xlsx, {file_stem}.xls, or {file_stem}.csv in {data_dir.resolve()}"
    )

def load_drone_delivery_data(data_dir: str | Path = "../data/raw"):
    train = load_table("train", data_dir)
    test = load_table("test", data_dir)
    sample_submission = load_table("sample_submission", data_dir)
    return train, test, sample_submission
