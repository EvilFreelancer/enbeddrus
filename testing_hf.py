# from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# model_name_or_path = "bert-base-multilingual-uncased"
# model_name_or_path = "Snowflake/snowflake-arctic-embed-m"
# model_name_or_path = "mixedbread-ai/mxbai-embed-large-v1"
# model_name_or_path = "output/enbeddrus-en-ru-2024-05-19_18-46-49"  # domain
model_name_or_path = "output/enbeddrus-en-ru-2024-05-20_11-30-48"  # parallel


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
sentences = [
    "PHP является скриптовым языком программирования, широко используемым для веб-разработки.",
    "PHP is a scripting language widely used for web development.",
    "PHP поддерживает множество баз данных, таких как MySQL, PostgreSQL и SQLite.",
    "PHP supports many databases like MySQL, PostgreSQL, and SQLite.",
    "Функция echo в PHP используется для вывода текста на экран.",
    "The echo function in PHP is used to output text to the screen.",
    "Машинное обучение помогает создавать интеллектуальные системы.",
    "Machine learning helps to create intelligent systems.",
]

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModel.from_pretrained(model_name_or_path)

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling. In this case, mean pooling.
embeddings = mean_pooling(model_output, encoded_input['attention_mask'])


# Функция для вычисления косинусного сходства
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# Сравнение эмбеддингов
for i in range(0, len(sentences), 2):
    sim = cosine_similarity(embeddings[i], embeddings[i + 1])
    print(f"{sim:.4f}\t'{sentences[i]}' | '{sentences[i + 1]}'")

# Вывод дополнительных сравнений
print("\n================================")
for i in range(len(sentences)):
    # for j in range(i + 1, len(sentences)):
    for j in range(len(sentences)):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        print(f"{sim:.4f}\t'{sentences[i]}' | '{sentences[j]}'")
