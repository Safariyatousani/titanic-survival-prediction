import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "titanic.csv"

def load_data(path: Path = DATA_PATH):
  return pd.read_csv(path)