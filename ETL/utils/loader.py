# ETL/utils/loader.py

from pathlib import Path

def load_to_csv_by_region(df):

    print("Loading data to CSV...")

    export_dir = Path("ETL") / "csv_exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0

    # ⭐ Fast grouping (Best Performance)
    for (region, lang), group_df in df.groupby(["region", "lang"]):

        filename = export_dir / f"{region.lower()}_{lang.lower()}_output.csv"

        # mode='w' — always overwrite; never append duplicate rows on re-runs
        group_df.to_csv(
            filename,
            index=False,
            encoding="utf-8-sig",
            mode="w",
        )

        row_count = len(group_df)
        total_rows += row_count

        action = "overwritten" if filename.exists() else "created"
        print(f"{region} | {lang} → {row_count} rows → {filename} [{action}]")

    print(f"\nTotal rows exported: {total_rows}")