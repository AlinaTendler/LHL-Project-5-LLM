# LLM Project

## Project Task
Topic classification of forum-style posts into 20 distinct categories using classical ML (TF-IDF + Logistic Regression) and Transformer-based models (DistilBERT).

## Dataset
- **Name:** 20 Newsgroups
- **Source:** Hugging Face (`SetFit/20_newsgroups`)
- **Structure:** ~11,000 training samples and ~7,500 test samples across 20 newsgroup categories.
- **Use Case:** A well-known benchmark dataset for text classification and topic modeling tasks.

## Pre-trained Model
- **Selected Model:** `distilbert-base-uncased`
- **Why:** 
  - Lightweight yet effective for text classification
  - Pre-trained on a large corpus
  - Compatible with Hugging Face `Trainer` API
  - Faster to fine-tune than full BERT

## Performance Metrics
| Model                       | Vectorization | Accuracy |
|----------------------------|---------------|----------|
| Logistic Regression        | TF-IDF        | ~75%     |
| Logistic Regression        | BoW           | ~72%     |
| Fine-tuned DistilBERT      | Tokenized     | ✅ ~82%  |

- Primary metric: **Accuracy**
- Evaluation done using Hugging Face's `evaluate` module

## Hyperparameters
- **Learning Rate:** 2e-5
- **Epochs:** 1
- **Batch Size:** 4 (due to memory limitations on macOS MPS backend)
- **Tokenizer:** `AutoTokenizer` from Hugging Face
- **Trainer API Settings:** `push_to_hub=True`, `logging_steps=10`, `no_cuda=True` (for CPU training on macOS)

## Relevant Links
- 🔗 [Model on Hugging Face Hub](https://huggingface.co/your-username/finetuned-newsgroups-distilbert)
- 📂 [20 Newsgroups Dataset](https://huggingface.co/datasets/SetFit/20_newsgroups)
