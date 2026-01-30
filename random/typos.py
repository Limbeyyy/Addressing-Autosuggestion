import pandas as pd

# Load your CSV
df = pd.read_csv("C:\\Users\\Yukesh Dhakal\\OneDrive\\Documents\\hints_separated.csv")   # change file name

# Select typo columns dynamically
typo_cols = [col for col in df.columns if col.startswith("typo_")]

# Convert wide → long format
df_long = df.melt(
    id_vars=["label"],      # keep label column
    value_vars=typo_cols,   # all typo columns
    value_name="typo",
    var_name="typo_column"
)

# Remove empty cells
df_long = df_long.dropna(subset=["typo"])

# Keep only needed columns
df_long = df_long[["typo", "label"]]

# Save new CSV
df_long.to_csv("typos.csv", index=False)

print("Done! Saved as typos.csv")
print(df_long.head())
