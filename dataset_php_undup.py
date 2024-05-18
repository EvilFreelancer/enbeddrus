import pandas as pd

input_csv = './dataset/docs_php.clean.csv'
output_csv = './dataset/docs_php.undup.csv'

df = pd.read_csv(input_csv)

# Only English
df = df[df['English'] != df['Russian']]

# Remove rows less than 10
df = df[df['English'].str.len() > 15]
# df = df[len(df['English']) > ]

# Order by column name
df = df.sort_values('English')

# Add index
df.index.name = 'id'
df.reset_index(inplace=True)

# Save
df.to_csv(output_csv, index=False)

print(f"Cleaned CSV saved to {output_csv}")
