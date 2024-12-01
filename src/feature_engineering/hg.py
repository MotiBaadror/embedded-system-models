from transformers import AutoTokenizer, AutoModel
import torch

# Load pre-trained tokenizer and model
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Sample text
text = "AI is revolutionizing the tech industry."

def get_embedding(text):
# Tokenize and get embeddings
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    try:
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # Mean pooling
        return True, embeddings
    except:
        return False

    # print(f"Embedding shape: {embeddings.shape}")
    # print(f"Embedding: {embeddings[0, :5]}")  # Show first 5 elements
