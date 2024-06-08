import pandas as pd

input_csv = './datasets/docs_go.clean.csv'
output_csv = './datasets/docs_go.undup.csv'

df = pd.read_csv(input_csv)

# Only English
df = df[df['en'] != df['ru']]

# Remove rows less than 10
df = df[df['en'].str.len() > 15]

# Order by column name
df = df.sort_values('en')
df.drop(['ID'], axis=1, inplace=True)

# Add index
df.index.name = 'id'
df.reset_index(inplace=True)

# Save
df.to_csv(output_csv, index=False)

print(f"Cleaned CSV saved to {output_csv}")
