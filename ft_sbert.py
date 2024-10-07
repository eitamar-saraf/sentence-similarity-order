import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sentence_transformers.losses import OnlineContrastiveLoss
from sentence_transformers.losses import SiameseDistanceMetric

# Load reference dialogues and summaries
dialogue_data = pd.read_csv('reference_dialogues.csv')
summary_data = pd.read_csv('reference_summaries.csv')

# Merge dialogues and summaries on id and dialog_id
merged_data = pd.merge(dialogue_data, summary_data, left_on='id', right_on='dialog_id')

# Create positive and negative pairs
positive_pairs = []
negative_pairs = []

for dialog_id in merged_data['dialog_id'].unique():
    # Get all the summary chunks for this dialogue
    dialogue_subset = merged_data[merged_data['dialog_id'] == dialog_id]
    dialogue_text = dialogue_subset['dialogue'].values[0]
    
    # Create positive pairs (conversation with its correct summary pieces)
    for _, row in dialogue_subset.iterrows():
        positive_pairs.append((dialogue_text, row['summary_piece'], 1))
    
    # Create negative pairs (conversation with random incorrect summary pieces)
    incorrect_summaries = merged_data[merged_data['dialog_id'] != dialog_id]['summary_piece'].sample(len(dialogue_subset))
    for incorrect_summary in incorrect_summaries:
        negative_pairs.append((dialogue_text, incorrect_summary, 0))

# Combine positive and negative pairs
pairs = positive_pairs + negative_pairs

# Create InputExample objects for training
training_data = [
    InputExample(texts=[conv, summary], label=label) 
    for conv, summary, label in pairs
]

# Split the data into training and validation sets (90% training, 10% validation)
train_data, val_data = train_test_split(training_data, test_size=0.1, random_state=42)

# Load pre-trained Sentence-BERT model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Create DataLoader for training and validation sets
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16)
val_dataloader = DataLoader(val_data, shuffle=False, batch_size=16)

# Use OnlineContrastiveLoss with cosine distance metric
train_loss = OnlineContrastiveLoss(
    model=model,
    distance_metric=SiameseDistanceMetric.COSINE_DISTANCE,  # Use cosine distance as the metric
    margin=0.5  # Margin between positive and negative pairs
)

# Fine-tune the model with validation
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=None,  # You can add a SentenceEvaluator here for validation metrics
    epochs=3,
    warmup_steps=100,
    show_progress_bar=True,
    evaluation_steps=100,  # Evaluate on validation data every 100 steps
    output_path='fine_tuned_sbert_online_contrastive'  # Save the model to this path
)

# Evaluate on the validation set after training
val_loss = model.evaluate(val_dataloader)
print(f"Validation loss: {val_loss}")

print("Model fine-tuned with OnlineContrastiveLoss and cosine distance, and validation evaluated.")


# Save the fine-tuned model
model.save('fine_tuned_sbert_online_contrastive')

print("Model fine-tuned with OnlineContrastiveLoss and cosine distance and saved successfully.")
