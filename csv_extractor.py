# main.py
from ETL.utils.extractor import extract_data
from ETL.utils.transformer import transform_data
from ETL.utils.loader import load_to_csv_by_region

def main():
    query = "SELECT input, label, rank_score, region, lang FROM feedback;"  
    print("Extracting data from database...")
    df = extract_data(query)
    
    if df.empty:
        print("No data found. Exiting.")
        return
    
    print("Transforming data...")
    df = transform_data(df)
    
    print("Loading data to CSV...")
    load_to_csv_by_region(df)

if __name__ == "__main__":
    main()