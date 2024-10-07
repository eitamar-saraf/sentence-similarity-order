import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# Load the fine-tuned SentenceTransformer model
model = SentenceTransformer('fine_tuned_sbert_online_contrastive')

# Load the data
dialogues = pd.read_csv('dialogues.csv')  # Contains 1400 chat conversations
summary_pieces = pd.read_csv('summary_pieces.csv')  # Contains 3996 shuffled pieces of summaries

# Extract the conversation texts and summary pieces
conversations = dialogues['dialogue'].tolist()  # List of chat conversations
conversation_ids = dialogues['id'].tolist()  # List of conversation IDs
summary_chunks = summary_pieces['summary_piece'].tolist()  # List of summary pieces

# Step 1: Embed all the conversations and summary pieces using the fine-tuned model
conversation_embeddings = model.encode(conversations, convert_to_tensor=True)
summary_embeddings = model.encode(summary_chunks, convert_to_tensor=True)

# Step 2: Compute cosine similarities between each conversation and each summary piece
cosine_scores = util.pytorch_cos_sim(conversation_embeddings, summary_embeddings)

# Step 3: For each conversation, find the most similar summary piece
best_matches = torch.argmax(cosine_scores, dim=1).tolist()

# Step 4: Map the best summary pieces back to the corresponding conversations with IDs
matched_summaries = [summary_chunks[idx] for idx in best_matches]
matched_summary_ids = [summary_pieces['id'].iloc[idx] for idx in best_matches]  # Corresponding summary IDs

# Create a DataFrame to store the results
result_df = pd.DataFrame({
    'conversation_id': conversation_ids,
    'conversation': conversations,
    'matched_summary_id': matched_summary_ids,
    'matched_summary': matched_summaries
})

# Save the results to a CSV file
result_df.to_csv('matched_conversations_with_summaries.csv', index=False)

print("Matched conversations with summary pieces have been saved.")
