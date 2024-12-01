from transformers import AutoTokenizer
model_name_or_path = "albert-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)


def preprocess_batch(example_batch, max_seq_length=128):
    texts_or_text_pairs = example_batch['text']
    features = tokenizer.batch_encode_plus(
                texts_or_text_pairs, max_length=max_seq_length, pad_to_max_length=True, truncation=True
    )
    return features