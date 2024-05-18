import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

# model_name_or_path = "./output/model_domain-en-ru-2024-05-18_14-12-55"
model_name_or_path = "Snowflake/snowflake-arctic-embed-xs"

model = SentenceTransformer(model_name_or_path)

sentences = []

# Получение эмбеддингов
embeddings = model.encode(sentences)


# Функция для вычисления косинусного сходства
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# Сравнение эмбеддингов
for i in range(0, len(sentences), 2):
    sim = cosine_similarity(embeddings[i], embeddings[i + 1])
    print(f"\n'{sentences[i]}'\n'{sentences[i + 1]}': {sim:.4f}\n")

# Вывод дополнительных сравнений
print("Дополнительные сравнения:")
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        print(f"\n'{sentences[i]}'\n'{sentences[j]}': {sim:.4f}")
