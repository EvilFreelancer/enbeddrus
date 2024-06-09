from sentence_transformers import SentenceTransformer, InputExample, LoggingHandler, models, evaluation, losses
from torch.utils.data import DataLoader
from sentence_transformers.datasets import ParallelSentencesDataset
from datetime import datetime
from datasets import load_dataset

import os
import logging
import sentence_transformers.util
import pandas as pd
import gzip
from tqdm.autonotebook import tqdm
import numpy as np
import zipfile
import io

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
logger = logging.getLogger(__name__)

# teacher_model_name = "Snowflake/snowflake-arctic-embed-xs"
# student_model_name = './output/model_domain'

# teacher_model_name = "intfloat/multilingual-e5-large"
# student_model_name = "intfloat/multilingual-e5-large"

# teacher_model_name = "Snowflake/snowflake-arctic-embed-xs"
# student_model_name = "Snowflake/snowflake-arctic-embed-xs"

# teacher_model_name = "Snowflake/snowflake-arctic-embed-s"
# student_model_name = "Snowflake/snowflake-arctic-embed-s"

# teacher_model_name = "Snowflake/snowflake-arctic-embed-m"
# student_model_name = "Snowflake/snowflake-arctic-embed-m"

# teacher_model_name = "mixedbread-ai/mxbai-embed-large-v1"
# student_model_name = "mixedbread-ai/mxbai-embed-large-v1"

teacher_model_name = 'bert-base-multilingual-uncased'
student_model_name = 'bert-base-multilingual-uncased'
# student_model_name = './output/enbeddrus_domain'

max_seq_length = 512  # Student model max. lengths for inputs (number of word pieces)
train_batch_size = 64  # Batch size for training
inference_batch_size = 64  # Batch size at inference
max_sentences_per_language = 500000  # Maximum number of  parallel sentences for training
train_max_sentence_length = 512  # Maximum length (characters) for parallel training sentences

num_epochs = 20  # Train for x epochs
num_warmup_steps = 10000  # Warmup steps

num_evaluation_steps = 100  # Evaluate performance after every xxxx steps
dev_sentences = 1000  # Number of parallel sentences to be used for development

# Define the language codes you would like to extend the model to
source_languages = set(["en"])  # Our teacher model accepts English (en) sentences
target_languages = set(["ru"])  # We want to extend the model to these new languages.

output_path = (
        "output/enbeddrus-"
        + "-".join(sorted(list(source_languages)) + sorted(list(target_languages)))
        + "-"
        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)


# This function downloads a corpus if it does not exist
def download_corpora(filepaths):
    if not isinstance(filepaths, list):
        filepaths = [filepaths]

    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(filepath, "does not exists. Try to download from server")
            filename = os.path.basename(filepath)
            url = "https://sbert.net/datasets/" + filename
            sentence_transformers.util.http_get(url, filepath)


def read_datasets():
    data = []

    # Read cleaned GoLang en&ru dataset
    docs_go_dataset = load_dataset("evilfreelancer/golang-en-ru")
    for item in docs_go_dataset['train']:
        src_text = item["en"].strip()
        trg_text = item["ru"].strip()
        if src_text and trg_text:
            data.append((src_text, trg_text))

    # Read cleaned OPUS PHP v1 en&ru dataset
    docs_php_dataset = load_dataset("evilfreelancer/opus-php-en-ru-cleaned")
    for item in docs_php_dataset['train']:
        src_text = item["English"].strip()
        trg_text = item["Russian"].strip()
        if src_text and trg_text:
            data.append((src_text, trg_text))

    # Read OPUS Books v1 en&ru dataset
    opus_dataset = load_dataset("Helsinki-NLP/opus_books", "en-ru")
    for item in opus_dataset['train']:
        src_text = item['translation']['en'].strip()
        trg_text = item['translation']['ru'].strip()
        if src_text and trg_text:
            data.append((src_text, trg_text))

    return data


def prepare_datasets(parallel_sentences_folder, dev_sentences):
    data = read_datasets()

    # Split data into train and dev sets
    train_files = []
    dev_files = []
    files_to_create = []
    os.makedirs(parallel_sentences_folder, exist_ok=True)

    for source_lang in source_languages:
        for target_lang in target_languages:
            output_filename_train = os.path.join(
                parallel_sentences_folder, "talks-{}-{}-train.tsv.gz".format(source_lang, target_lang)
            )
            output_filename_dev = os.path.join(
                parallel_sentences_folder, "talks-{}-{}-dev.tsv.gz".format(source_lang, target_lang)
            )
            train_files.append(output_filename_train)
            dev_files.append(output_filename_dev)
            if not os.path.exists(output_filename_train) or not os.path.exists(output_filename_dev):
                files_to_create.append(
                    {
                        "src_lang": source_lang,
                        "trg_lang": target_lang,
                        "fTrain": gzip.open(output_filename_train, "wt", encoding="utf8"),
                        "fDev": gzip.open(output_filename_dev, "wt", encoding="utf8"),
                        "devCount": 0,
                    }
                )

    for src_text, trg_text in tqdm(data, desc="Sentences"):
        for outfile in files_to_create:
            if outfile["devCount"] < dev_sentences:
                outfile["devCount"] += 1
                fOut = outfile["fDev"]
            else:
                fOut = outfile["fTrain"]

            fOut.write("{}\t{}\n".format(src_text, trg_text))

    for outfile in files_to_create:
        outfile["fTrain"].close()
        outfile["fDev"].close()

    return train_files, dev_files


# Here we define train and dev corpora
train_corpus = "./dataset/docs_php.undup.csv"
sts_corpus = "datasets/stsbenchmark.zip"
parallel_sentences_folder = "parallel-sentences/"

# Check if the file exists. If not, they are downloaded
download_corpora([sts_corpus])

# Prepare datasets
train_files, dev_files = prepare_datasets(parallel_sentences_folder, dev_sentences)

# Start the extension of the teacher model to multiple languages
logger.info("Load teacher model")
teacher_model = SentenceTransformer(teacher_model_name)

logger.info("Create student model from scratch")
word_embedding_model = models.Transformer(student_model_name, max_seq_length=max_seq_length)
# Apply mean pooling to get one fixed-sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
student_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Read Parallel Sentences Dataset
train_data = ParallelSentencesDataset(
    student_model=student_model, teacher_model=teacher_model, batch_size=inference_batch_size, use_embedding_cache=True
)
for train_file in train_files:
    train_data.load_data(
        train_file, max_sentences=max_sentences_per_language, max_sentence_length=train_max_sentence_length
    )

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MSELoss(model=student_model)

# Evaluate cross-lingual performance on different tasks
evaluators = []  # evaluators has a list of different evaluator classes we call periodically

for dev_file in dev_files:
    logger.info("Create evaluator for " + dev_file)
    src_sentences = []
    trg_sentences = []
    with gzip.open(dev_file, "rt", encoding="utf8") as fIn:
        for line in fIn:
            splits = line.strip().split("\t")
            if splits[0] != "" and splits[1] != "":
                src_sentences.append(splits[0])
                trg_sentences.append(splits[1])

    # Mean Squared Error (MSE) measures the (euclidean) distance between teacher and student embeddings
    dev_mse = evaluation.MSEEvaluator(
        src_sentences,
        trg_sentences,
        name=os.path.basename(dev_file),
        teacher_model=teacher_model,
        batch_size=inference_batch_size,
    )
    evaluators.append(dev_mse)

    # TranslationEvaluator computes the embeddings for all parallel sentences. It then checks if the embedding of
    # source[i] is the closest to target[i] out of all available target sentences
    dev_trans_acc = evaluation.TranslationEvaluator(
        src_sentences, trg_sentences, name=os.path.basename(dev_file), batch_size=inference_batch_size
    )
    evaluators.append(dev_trans_acc)

# Read cross-lingual Semantic Textual Similarity (STS) data
all_languages = list(set(list(source_languages) + list(target_languages)))
sts_data = {}

# Open the ZIP File of STS2017-extended.zip and check for which language combinations we have STS data
with zipfile.ZipFile(sts_corpus) as zip:
    filelist = zip.namelist()
    sts_files = []

    for i in range(len(all_languages)):
        for j in range(i, len(all_languages)):
            lang1 = all_languages[i]
            lang2 = all_languages[j]
            filepath = "STS2017-extended/STS.{}-{}.txt".format(lang1, lang2)
            if filepath not in filelist:
                lang1, lang2 = lang2, lang1
                filepath = "STS2017-extended/STS.{}-{}.txt".format(lang1, lang2)

            if filepath in filelist:
                filename = os.path.basename(filepath)
                sts_data[filename] = {"sentences1": [], "sentences2": [], "scores": []}

                fIn = zip.open(filepath)
                for line in io.TextIOWrapper(fIn, "utf8"):
                    sent1, sent2, score = line.strip().split("\t")
                    score = float(score)
                    sts_data[filename]["sentences1"].append(sent1)
                    sts_data[filename]["sentences2"].append(sent2)
                    sts_data[filename]["scores"].append(score)

for filename, data in sts_data.items():
    test_evaluator = evaluation.EmbeddingSimilarityEvaluator(
        data["sentences1"],
        data["sentences2"],
        data["scores"],
        batch_size=inference_batch_size,
        name=filename,
        show_progress_bar=False,
    )
    evaluators.append(test_evaluator)

# Train the model
student_model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: np.mean(scores)),
    epochs=num_epochs,
    warmup_steps=num_warmup_steps,
    evaluation_steps=num_evaluation_steps,
    output_path=output_path,
    save_best_model=True,
    optimizer_params={"lr": 2e-5, "eps": 1e-6},
)
