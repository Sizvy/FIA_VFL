import pandas as pd

# Read the CSV file
df = pd.read_csv('../../../datasets/drive_cleaned.csv')

# Remove duplicate columns (keeping the first occurrence)
# Method 1: Using transpose and drop_duplicates (more efficient for many columns)
unique_df = df.T.drop_duplicates().T

# Alternative Method 2: Using a dictionary to identify unique columns
# unique_cols = {}
# for col in df.columns:
#     # Convert column to a tuple to use as dictionary key
#     col_tuple = tuple(df[col])
#     if col_tuple not in unique_cols:
#         unique_cols[col_tuple] = col
# unique_df = df[unique_cols.values()]

# Save the result to a new CSV file
unique_df.to_csv('../../../datasets/drive_cleaned_unique_columns.csv', index=False)

print(f"Original shape: {df.shape}")
print(f"After removing duplicate columns: {unique_df.shape}")
print(f"Number of duplicate columns removed: {df.shape[1] - unique_df.shape[1]}")
