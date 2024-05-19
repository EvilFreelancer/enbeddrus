import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

model_name_or_path = "../output/model_domain-en-ru-2024-05-19_12-16-02"
# model_name_or_path = "Snowflake/snowflake-arctic-embed-s"

model = SentenceTransformer(model_name_or_path)

sentences = [
    "PHP является скриптовым языком программирования, широко используемым для веб-разработки.",
    "PHP is a scripting language widely used for web development.",
    "PHP поддерживает множество баз данных, таких как MySQL, PostgreSQL и SQLite.",
    "PHP supports many databases like MySQL, PostgreSQL, and SQLite.",
    "Функция echo в PHP используется для вывода текста на экран.",
    "The echo function in PHP is used to output text to the screen.",
    "Машинное обучение помогает создавать интеллектуальные системы.",
    "Machine learning helps to create intelligent systems.",
    "Велосипедные прогулки полезны для здоровья и экологии.",
    "Cycling is beneficial for health and the environment.",
    "Python является одним из самых популярных языков программирования в мире.",
    "Python is one of the most popular programming languages in the world.",
    "Живопись успокаивает и развивает креативность.",
    "Painting is calming and fosters creativity.",
    "Базы данных хранят и управляют большими объемами данных.",
    "Databases store and manage large volumes of data.",
    "Чтение книг обогащает словарный запас и расширяет кругозор.",
    "Reading books enriches vocabulary and broadens horizons.",
    "Алгоритмы важны для решения сложных задач в компьютерных науках.",
    "Algorithms are crucial for solving complex problems in computer science.",
    "Музыка способна влиять на настроение и эмоции человека.",
    "Music can affect a person's mood and emotions.",
    "Интернет вещей (IoT) соединяет различные устройства в единую сеть.",
    "The Internet of Things (IoT) connects various devices into a single network.",
    "Прогулки на природе помогают снять стресс и улучшить самочувствие.",
    "Walks in nature help reduce stress and improve well-being."
]

# Получение эмбеддингов
embeddings = model.encode(sentences)


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
    #for j in range(i + 1, len(sentences)):
    for j in range(len(sentences)):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        print(f"{sim:.4f}\t'{sentences[i]}' | '{sentences[j]}'")
