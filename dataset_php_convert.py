import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# input_csv = './datasets/docs_php.undup.csv'
# ds_df = pd.read_csv(input_csv)
# ds_json = ds_df[['id', 'English']]
# ds_json = ds_json.rename(columns={'English': 'text', 'id': '_id'})
# ds_json = ds_json.assign(title="")
# ds_json = ds_json.astype({'_id': 'string'})

dataset = load_dataset("evilfreelancer/opus-php-en-ru-cleaned", split='train')
ds_df = dataset.to_pandas()
ds_json = ds_df[['English']].rename(columns={'English': 'text'})
ds_json['_id'] = ds_df.index.astype(str)
ds_json = ds_json.assign(title="")

ds_train, ds_test = train_test_split(ds_json, test_size=0.3)
ds_train.to_json('./datasets/corpus.jsonl', orient='records', lines=True)  # Save training dataset
ds_test.to_json('./datasets/queries.jsonl', orient='records', lines=True)  # Save evaluation dataset
