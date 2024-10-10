from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import InputExample, SentenceTransformer, losses, evaluation, util
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from nltk.tokenize import sent_tokenize


def match_chunks_with_summaries_pieces(chunked_dialogues: pd.DataFrame, summary_pieces: pd.DataFrame, model: SentenceTransformer) -> pd.DataFrame:
    """
    Match each chunk in the dialog to it's summerization piece.
    Each chunk can have several summary pieces, or none at all.
    Pay attention, we have the chronological order summary pieces (position_index). position_index 0 can not be matched with chunk 1, while position_index 1 matches with chunk 0. 
    So we need to validate the 'position_index' field.
    Good Example:
    we have 3 chunks and 3 summary pieces:
    summary_piece 0 -> chunk 0 
    summary_piece 1 -> chunk 0
    summary_piece 2 -> chunk 1

    Good Example:
    we have 3 chunks and 3 summary pieces:
    summary_piece 0 -> chunk 2
    summary_piece 1 -> chunk 2
    summary_piece 2 -> chunk 2

    Bad Example:
    we have 3 chunks and 3 summary pieces:
    summary_piece 0 -> chunk 1
    summary_piece 1 -> chunk 0
    summary_piece 2 -> chunk 2

    Explanation:
    In the bad example, summary_piece 0 is matched with chunk 1, when summary_piece 1 is matched with chunk 0. This is incorrect because the summary pieces are in chronological order.

    Bad Example:
    we have 3 chunks and 3 summary pieces:
    summary_piece 0 -> chunk 1
    summary_piece 1 -> chunk 1
    summary_piece 2 -> chunk 0

    Explanation:
    In the bad example, summary_piece 0 is matched with chunk 1, and summary_piece 1 is matched with chunk 1 as well. While summary_piece 2 is matched with chunk 0. This is incorrect because the summary pieces are in chronological order.


    Args:
    chunked_dialogues (pd.DataFrame): The DataFrame containing the dialogue chunks. It should have columns 'id', 'sub_id', and 'chunk'
    summary_pieces (pd.DataFrame): The DataFrame containing the summary pieces. It should have columns 'dialog_id', 'summary_piece', and 'position_index'
    model (SentenceTransformer): The SentenceTransformer model for encoding text

    Returns:
    pd.DataFrame: A DataFrame containing the matched chunks and summary pieces
    """
    grouped_dialogues = chunked_dialogues.groupby('id')

    matched_data = []

    for dialog_id, group in grouped_dialogues:
        # Get the summary pieces for this dialogue
        summary_subset = summary_pieces[summary_pieces['dialog_id'] == dialog_id]
        summary_subset = summary_subset.sort_values('position_index')

        # Get number of chunks
        num_chunks = len(group)
        if num_chunks == 0:
            continue

        elif num_chunks == 1:
            # If there is only one chunk, match it with all the summary pieces
            for _, row in summary_subset.iterrows():
                chunk_text = group['chunk'].values[0]
                summary_text = row['summary_piece']
                matched_data.append({'id': dialog_id, 'sub_id': 0, 'chunk': chunk_text, 'summary_piece': summary_text, 'position_index': row['position_index']})

        else:
            group = group.sort_values('sub_id')

            # Extract the chunk texts and compute their embeddings
            chunk_texts = group['chunk'].tolist()
            chunk_embeddings = model.encode(chunk_texts)

            # Extract the summary piece texts and compute their embeddings
            summary_texts = summary_subset['summary_piece'].tolist()
            summary_embeddings = model.encode(summary_texts)

            # Initialize the last assigned chunk index
            last_assigned_chunk_index = 0

            # Loop over each summary piece in order of 'position_index'
            for i, summary_embedding in enumerate(summary_embeddings):
                # Consider chunks from the last assigned chunk index onwards
                chunk_indices = list(range(last_assigned_chunk_index, num_chunks))

                # Compute similarities between the current summary piece and the remaining chunks
                similarities = cosine_similarity([summary_embedding], chunk_embeddings[chunk_indices])[0]

                # Find the chunk with the highest similarity
                max_sim_index = similarities.argmax()
                assigned_chunk_index = chunk_indices[max_sim_index]

                # Ensure that the assigned chunk index is not less than the last assigned index
                assigned_chunk_index = max(assigned_chunk_index, last_assigned_chunk_index)

                # Get the corresponding chunk text
                chunk_text = chunk_texts[assigned_chunk_index]

                # Get the summary text and position index
                summary_text = summary_texts[i]
                position_index = summary_subset.iloc[i]['position_index']

                # Append the matched data
                matched_data.append({'id': dialog_id, 'sub_id': group.iloc[assigned_chunk_index]['sub_id'], 
                                     'chunk': chunk_text, 'summary_piece': summary_text, 'position_index': position_index
                                     })
                
                # Update the last assigned chunk index
                last_assigned_chunk_index = assigned_chunk_index 

    return pd.DataFrame(matched_data)
        




def split_dialogue_into_chunks(dialogue:str, model:SentenceTransformer, chunk_size:int=32, overlap:int=0) -> list:
    tokens = model.tokenizer.tokenize(dialogue, truncation=False)
    chunks = []
    start = 0
    min_chunk_size = chunk_size // 2 # Define a minimum acceptable chunk size

    while start < len(tokens):
        end = start + chunk_size

        if end >= len(tokens):
            # Handle the last chunk
            remaining_tokens = len(tokens) - start
            if remaining_tokens < min_chunk_size and len(chunks) > 0:
                # Adjust the start index to include more tokens from the previous chunk
                new_start = max(0, len(tokens) - min_chunk_size)
                chunk_tokens = tokens[new_start:len(tokens)]
                chunk_text =  model.tokenizer.convert_tokens_to_string(chunk_tokens)
                chunks.append(chunk_text)
                break  # Exit the loop after adjusting the last chunk
            else:
                # If it's the first chunk or the remaining tokens are sufficient
                chunk_tokens = tokens[start:len(tokens)]
                chunk_text =  model.tokenizer.convert_tokens_to_string(chunk_tokens)
                chunks.append(chunk_text)
                break  # Exit the loop after adding the last chunk
        else:
            # Process regular chunks
            chunk_tokens = tokens[start:end]
            chunk_text =  model.tokenizer.convert_tokens_to_string(chunk_tokens)
            chunks.append(chunk_text)
            start += (chunk_size - overlap)

    return chunks



def chunk_dialog(dialogues: pd.DataFrame, model: SentenceTransformer) -> pd.DataFrame:
    chunked_dialogues = []
    for i, row in dialogues.iterrows():
        dialogue_id = row['id']
        dialogue_text = row['dialogue']
        # Split the dialogue into chunks
        chunks = split_dialogue_into_chunks(dialogue_text, model)
        # Append each chunk to the new data list with a sub_id
        for i, chunk in enumerate(chunks):
            chunked_dialogues.append({'id': dialogue_id, 'sub_id': i, 'chunk': chunk})
    return pd.DataFrame(chunked_dialogues)


def prepare_input_data(data: pd.DataFrame) -> List[InputExample]:
    examples = []
    for idx, row in data.iterrows():
        text_anchor = row['chunk']
        text_positive = row['summary_piece']
        examples.append(InputExample(texts=[text_anchor, text_positive]))
    return examples


def prepare_data_for_evaluation(data: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Set[str]]]:
    corpus = {}
    queries = {}
    relevant_docs = {}

    for idx, row in data.iterrows():
        query_id = f"query_{row['id']}_{row['position_index']}"
        query_text = row['summary_piece']

        doc_id = f"doc_{row['id']}_{row['sub_id']}"
        doc_text = row['chunk']

        if query_id not in queries:
            queries[query_id] = query_text
        
        if doc_id not in corpus:
            corpus[doc_id] = doc_text
        
        if query_id not in relevant_docs:
            relevant_docs[query_id] = set()

            # Add all docs from the same conversation as relevant
            docs_id = data[data['id'] == row['id']]['sub_id'].unique()
            relevant_docs[query_id].update([f"doc_{row['id']}_{doc_id}" for doc_id in docs_id])
    
    return corpus, queries, relevant_docs
    

def train(dialogues_path:Path, summary_pieces_path:Path):
    dialogues = pd.read_csv(dialogues_path)
    summary_pieces = pd.read_csv(summary_pieces_path)

    # Load the model
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    # Split the dialogues into chunks if they are too long
    chunked_dialogues = chunk_dialog(dialogues, model)

    # match the chunks with the summary pieces, use cosine similarity
    conv_chunk_with_summary_piece_match = match_chunks_with_summaries_pieces(chunked_dialogues, summary_pieces, model)
    

    unique_ids = conv_chunk_with_summary_piece_match['id'].unique()
    train_ids, val_ids = train_test_split(unique_ids, test_size=0.1, random_state=42)
    train_df = conv_chunk_with_summary_piece_match[conv_chunk_with_summary_piece_match['id'].isin(train_ids)]
    val_df = conv_chunk_with_summary_piece_match[conv_chunk_with_summary_piece_match['id'].isin(val_ids)]
    

    train_examples = prepare_input_data(train_df)
    # val_examples = prepare_input_data(val_df)

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=64)

    train_loss = losses.MultipleNegativesRankingLoss(model)

    corpus, queries, relevant_docs = prepare_data_for_evaluation(val_df)
    ir_evaluator = evaluation.InformationRetrievalEvaluator(
        queries, corpus, relevant_docs,
        show_progress_bar=True,
        batch_size=16,
        name='ir_evaluator',
        map_at_k=[1,3],
        ndcg_at_k=[1,3],
        mrr_at_k=[1,3],
        accuracy_at_k=[1,3],
        precision_recall_at_k=[1,3],
        main_score_function='cosine'
    )

    
    num_epochs = 3
    model_save_path = './trained_model'
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) 

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=ir_evaluator,
        epochs=num_epochs,
        evaluation_steps=10,
        warmup_steps=warmup_steps,
        output_path=model_save_path,
        show_progress_bar=True,
        save_best_model=True
    )


def match_summary_pieces(dialogues_df: pd.DataFrame, summary_pieces_df: pd.DataFrame) -> pd.DataFrame:
    # Load the model from the saved path
    model = SentenceTransformer('trained_model')

    # Split the dialogues into chunks if they are too long
    chunked_dialogues = chunk_dialog(dialogues_df, model)

    # Encode conversation chunks
    conversation_chunks = chunked_dialogues['chunk'].tolist()
    conversation_embeddings = model.encode(conversation_chunks, convert_to_tensor=True) # (num_chunks, embedding_dim)

    # Encode summary pieces
    summary_pieces = summary_pieces_df['summary_piece'].tolist()
    summary_embeddings = model.encode(summary_pieces, convert_to_tensor=True) # (num_summary_pieces, embedding_dim)

    # Compute cosine similarity between conversation chunks and summary pieces
    similarity_matrix = util.cos_sim(summary_embeddings, conversation_embeddings) # (num_summary_pieces, num_chunks)

    # For each summary piece, find the index of the most similar conversation chunk
    max_similarities, matched_indices = torch.max(similarity_matrix, dim=1)

    # Convert indices and similarities to NumPy arrays
    matched_indices = matched_indices.cpu().numpy()
    max_similarities = max_similarities.cpu().numpy()

    # Create a DataFrame to map summary pieces to conversation chunks
    df_mapping = pd.DataFrame({
        'summary_piece': summary_pieces_df['summary_piece'],
        'matched_chunk_index': matched_indices,
        'similarity': max_similarities
    })

    # Reset index to align with 'matched_chunk_index'
    chunked_dialogues = chunked_dialogues.reset_index()

    # Merge to get 'dialog_id' and 'sub_id' from conversation DataFrame
    df_mapping = df_mapping.merge(
        chunked_dialogues[['index', 'id', 'sub_id']],
        left_on='matched_chunk_index',
        right_on='index',
        how='left'
    )

    df_mapping.rename(columns={'id': 'dialog_id'}, inplace=True)

    # Sort by 'dialog_id' and 'sub_id' to order chronologically
    df_mapping.sort_values(by=['dialog_id', 'sub_id'], inplace=True)

    # Identify dialog_ids with duplicate sub_id's
    duplicate_sub_ids = df_mapping[df_mapping.duplicated(subset=['dialog_id', 'sub_id'], keep=False)]
    duplicate_dialog_ids = duplicate_sub_ids['dialog_id'].unique()

    # Process each dialog_id separately
    results = []
    grouped = df_mapping.groupby('dialog_id')

    for dialog_id, group in grouped:
        if dialog_id not in duplicate_dialog_ids:
            # No duplicates, assign position_index directly
            group = group.copy()
            group['position_index'] = range(0, len(group))
            results.append(group)
        
        else:
            # Duplicates exist, need to split conversation chunks into sentences
            group = group.copy()
            # Get conversation chunks for this dialog_id
            dialogue = dialogues_df[dialogues_df['id'] == dialog_id]

            # Split each dialogue into 24 tokens each chunk
            sentences = split_dialogue_into_chunks(dialogue.values[0], model, chunk_size=24, overlap=0)

            sentence_embeddings = model.encode(sentences, convert_to_tensor=True) # (num_sentences, embedding_dim)

            summary_pieces_group = group['summary_piece'].tolist()  # (num_summary_pieces)
            summary_embeddings_group = model.encode(summary_pieces_group, convert_to_tensor=True) # (num_summary_pieces, embedding_dim)

            # Compute cosine similarity between sentences and summary pieces
            similarity_matrix = util.cos_sim(summary_embeddings_group, sentence_embeddings) # (num_summary_pieces, num_sentences)

            # For each summary piece, find the index of the most similar sentence
            max_similarities, matched_sentence_indices  = torch.max(similarity_matrix, dim=1)
            matched_positions = matched_sentence_indices.cpu().numpy()

            # Add the matched sentence indices to the group
            group['matched_sentence_index'] = matched_positions

            # Sort the group by 'sentence_sub_id' and 'matched_sentence_index' to order chronologically
            group.sort_values(by=['matched_sentence_index'], inplace=True)

            # Assign position_index within this group
            group['position_index'] = range(0, len(group))
            results.append(group)

    # Combine all groups
    final_df = pd.concat(results, ignore_index=True)

    return final_df


def main(dialog_path:str, summary_pieces_path:str, output_path:str) -> None:
    dialog_path = Path(dialog_path)
    summary_path = Path(summary_pieces_path)
    output_path = Path(output_path)

    if not dialog_path.exists():
        raise FileNotFoundError(f"File not found: {dialog_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"File not found: {summary_path}")
    
    dialogues = pd.read_csv(dialog_path)
    summary_pieces = pd.read_csv(summary_path)

    output_df = match_summary_pieces(dialogues, summary_pieces)
    output_df = output_df[['dialog_id', 'summary_piece', 'position_index']]
    output_df.to_csv(output_path, index=False)


if __name__ == '__main__':
    # train_dialogues_path, train_summary_pieces_path = Path('reference_dialogues.csv'), Path('reference_summaries.csv')
    # train(train_dialogues_path, train_summary_pieces_path)

    test_dialogues_path, test_summary_pieces_path = 'dialogues.csv', 'summary_pieces.csv'
    output_path = 'attributed_summary_pieces.csv'
    main(test_dialogues_path, test_summary_pieces_path, output_path)