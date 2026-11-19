from ucimlrepo import fetch_ucirepo
from pathlib import Path
import pandas as pd

def main():
    print("Downloading UCI Metro Interstate Traffic Volume (ID 492)...")
    ds = fetch_ucirepo(id=492)
    df = pd.concat([ds.data.features.reset_index(drop=True), ds.data.targets.reset_index(drop=True)], axis=1)
    df.columns = [str(c).strip().lower() for c in df.columns]
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/traffic.csv", index=False)
    print(f"Saved {len(df):,} records to data/traffic.csv")
if __name__ == "__main__": main()
