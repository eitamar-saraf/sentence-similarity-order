


import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer
import torch

# Load conversation data and summary pieces
dialogues = pd.read_csv('dialogues.csv')
summary_pieces = pd.read_csv('summary_pieces.csv')

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_embeddings(text_list):
    # Tokenize and get embeddings for each text in the list
    encoded_inputs = tokenizer(text_list, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**encoded_inputs)
    return outputs.last_hidden_state.mean(dim=1)  # Average embeddings over sequence length

# Generate embeddings for dialogues and summary pieces
dialogue_embeddings = get_embeddings(dialogues['text'].tolist())
summary_embeddings = get_embeddings(summary_pieces['text'].tolist())

# Compute cosine similarity between each dialogue and each summary piece
similarities = cosine_similarity(summary_embeddings, dialogue_embeddings)

# Attribute each summary piece to the most similar dialogue
attributed_summary = pd.DataFrame(similarities, columns=dialogues['id'])
summary_pieces['attributed_dialogue_id'] = attributed_summary.idxmax(axis=1)

# Save attributed summary pieces
summary_pieces.to_csv('attributed_summary_pieces.csv', index=False)
