import pandas as pd
from transformers import BertForNextSentencePrediction, BertTokenizer
import torch

# Load the pre-trained BERT NSP model and tokenizer
nsp_model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the matched conversations with summary pieces
matched_df = pd.read_csv('matched_conversations_with_summaries.csv')

# Group the matched summary pieces by conversation
grouped = matched_df.groupby('conversation_id')

# Function to perform NSP task between two summary pieces
def get_nsp_score(summary_piece1, summary_piece2):
    encoding = tokenizer(summary_piece1, summary_piece2, return_tensors='pt')
    outputs = nsp_model(**encoding, labels=torch.LongTensor([1]))
    loss, logits = outputs[:2]
    return torch.softmax(logits, dim=1)[0][0].item()  # Probability that summary_piece2 follows summary_piece1

# Reorder summary pieces within each conversation using NSP
reordered_summaries = []

for conversation_id, group in grouped:
    summary_pieces = group['matched_summary'].tolist()
    
    # If there's only one summary piece, no need to reorder
    if len(summary_pieces) == 1:
        reordered_summaries.append((conversation_id, summary_pieces))
        continue
    
    # Pairwise comparisons of summary pieces using NSP
    pairwise_scores = {}
    for i, piece1 in enumerate(summary_pieces):
        for j, piece2 in enumerate(summary_pieces):
            if i != j:
                score = get_nsp_score(piece1, piece2)
                pairwise_scores[(i, j)] = score

    # Reorder based on the pairwise NSP scores (greedy approach)
    current_idx = 0
    ordered_pieces = [summary_pieces[current_idx]]
    
    while len(ordered_pieces) < len(summary_pieces):
        next_idx = max([(j, pairwise_scores[(current_idx, j)]) for j in range(len(summary_pieces)) if j not in ordered_pieces], key=lambda x: x[1])[0]
        ordered_pieces.append(summary_pieces[next_idx])
        current_idx = next_idx

    reordered_summaries.append((conversation_id, ordered_pieces))

# Save the reordered summaries
reordered_df = pd.DataFrame({
    'conversation_id': [conv_id for conv_id, _ in reordered_summaries],
    'reordered_summaries': [' '.join(summaries) for _, summaries in reordered_summaries]
})

# Save to CSV
reordered_df.to_csv('reordered_conversations_with_summaries.csv', index=False)

print("Reordered summaries saved to 'reordered_conversations_with_summaries.csv'.")