import pandas as pd
from sklearn.model_selection import train_test_split

input_csv = './dataset/docs_php.undup.csv'

ds_df = pd.read_csv(input_csv)

ds_json = ds_df[['id', 'English']]
ds_json = ds_json.rename(columns={'English': 'text', 'id': '_id'})
ds_json = ds_json.assign(title="")
ds_json = ds_json.astype({'_id': 'string'})

ds_train, ds_test = train_test_split(ds_json, test_size=0.3)
ds_train.to_json('./dataset/corpus.jsonl', orient='records', lines=True)  # Save training dataset
ds_test.to_json('./dataset/queries.jsonl', orient='records', lines=True)  # Save evaluation dataset
