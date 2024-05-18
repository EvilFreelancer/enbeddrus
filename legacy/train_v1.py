from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd

model_en = SentenceTransformer('intfloat/multilingual-e5-small')
model_ru = SentenceTransformer('intfloat/multilingual-e5-small')

dataset = pd.read_csv('dataset.csv')

examples_en = dataset['en']
examples_ru = dataset['ru']
train_batch_size = 2

labels_en_en = model_en.encode(examples_en)
examples_en_ru = [InputExample(texts=[x], label=labels_en_en[i]) for i, x in enumerate(examples_en)]
loader_en_ru = DataLoader(examples_en_ru, batch_size=train_batch_size)

examples_ru_ru = [InputExample(texts=[x], label=labels_en_en[i]) for i, x in enumerate(examples_ru)]
loader_ru_ru = DataLoader(examples_ru_ru, batch_size=train_batch_size)

train_loss = losses.MSELoss(model=model_ru)
model_ru.fit(
    [(loader_en_ru, train_loss), (loader_ru_ru, train_loss)],
    epochs=10,
    show_progress_bar=False
)
model_ru.save('model_ru')
