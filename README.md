# Enbedrus - ENglish and RUSsian emBEDDer

This is a BERT (uncased) [sentence-transformers](https://www.SBERT.net) model: It maps sentences & paragraphs to a 768 dimensional
dense vector space and can be used for tasks like clustering or semantic search.

- **Parameters**: 168 million
- **Layers**: 12
- **Hidden Size**: 768
- **Attention Heads**: 12
- **Vocabulary Size**: 119,547
- **Maximum Sequence Length**: 512 tokens

The Enbeddrus model is designed to extract similar embeddings for comparable English and Russian phrases. It is based on
the [bert-base-multilingual-uncased](https://huggingface.co/google-bert/bert-base-multilingual-cased) model and was
trained over 20 epochs on the following datasets:

- [evilfreelancer/opus-php-en-ru-cleaned](https://huggingface.co/datasets/evilfreelancer/opus-php-en-ru-cleaned) (train): 1.6k lines
- [evilfreelancer/golang-en-ru](https://huggingface.co/datasets/evilfreelancer/golang-en-ru) (train): 554 lines
- [Helsinki-NLP/opus_books](https://huggingface.co/datasets/Helsinki-NLP/opus_books/viewer/en-ru) (en-ru, train): 17.5k lines

The goal of this model is to generate identical or very similar embeddings regardless of whether the text is written in
English or Russian.

[Enbeddrus GGUF](https://ollama.com/evilfreelancer/enbeddrus) version available via Ollama.

## Envaluation test

Models tested via [encodechka](https://github.com/avidale/encodechka)


| Name                  | evilfreelancer/enbeddrus-v0.1 | evilfreelancer/enbeddrus-v0.1-domain | evilfreelancer/enbeddrus-v0.2 |
| --------------------- | ----------------------------- | ------------------------------------ | ----------------------------- |
| STSBTask              | 0.6418501890569303            | 0.6418501890569303                   | 0.6382642407246252            |
| ParaphraserTask       | 0.5396186809125094            | 0.5396186809125094                   | 0.5491558495250873            |
| XnliTask              | 0.37045908183632736           | 0.37045908183632736                  | 0.36666666666666664           |
| SentimentTask         | 0.7306666666666667            | 0.7306666666666667                   | 0.7246666666666667            |
| ToxicityTask          | 0.8923319999999999            | 0.8923319999999999                   | 0.894758                      |
| InappropriatenessTask | 0.7092166782043772            | 0.7092166782043772                   | 0.719323712657756             |
| IntentsTask           | 0.7086                        | 0.7162                               | 0.7128                        |
| IntentsXTask          | 0.5116                        | 0.46                                 | 0.5314                        |
| FactRuTask            | n/a                           | n/a                                  | n/a                           |
| RudrTask              | n/a                           | n/a                                  | n/a                           |
| SpeedTask (cuda)      | 4.313722451527913             | 4.339381853739421                    | 4.251763025919597             |
| SpeedTask (cpu)       | 34.0190052986145              | 34.990905125935875                   | 34.441959857940674            |
