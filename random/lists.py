import pandas as pd

# --- Step 1: Load your data ---
# Replace 'data.csv' with your file path and 'column_name' with the target column
df = pd.read_csv("data.csv", encoding="utf-8")

# --- Step 2: Remove duplicates ---
# Select the column you want unique values from
column_name = "target"
unique_values = df[column_name].astype(str).str.strip().drop_duplicates()

# --- Step 3: Optional: sort values ---
unique_values = unique_values.sort_values()

# --- Step 4: Save or print ---
print("Unique values:")
print(unique_values.tolist())  # print as list

# Optionally, save to a new CSV
unique_values.to_csv("unique_values.csv", index=False, header=True)
