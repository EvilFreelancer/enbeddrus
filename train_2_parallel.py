from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
from sentence_transformers import evaluation
import gzip
import csv

# Загрузка дообученной модели
model = SentenceTransformer('path_to_save_model/')

# Подготовка датасета с параллельными корпусами
train_examples = []
with gzip.open('parallel_corpus_en_ru.tsv.gz', 'rt', encoding='utf8') as fIn:
    reader = csv.reader(fIn, delimiter='\t')
    for row in reader:
        if row:
            eng_text = row[0]
            rus_text = row[1]
            train_examples.append(InputExample(texts=[eng_text, rus_text], label=1.0))

# DataLoader для обучения
train_dataloader = DataLoader(train_examples, batch_size=16, shuffle=True)

# Функция потерь MSE
train_loss = losses.MSELoss(model=model)
evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(train_examples, name='eval')
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=100,
    evaluator=evaluator,
    evaluation_steps=500
)
model.save('model_ru/')
