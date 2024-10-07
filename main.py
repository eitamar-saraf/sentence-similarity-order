from typing import Dict, List
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
from pathlib import Path



def split_dialogue_into_chunks(dialogue:str, model:SentenceTransformer, overlap:int=64) -> list:
    tokens = model.tokenizer.tokenize(dialogue, truncation=False,)
    chunk_size = model.get_max_seq_length()
    chunks = []
    start = 0
    min_chunk_size = chunk_size # Define a minimum acceptable chunk size

    while start < len(tokens):
        end = start + chunk_size

        if end >= len(tokens):
            # Handle the last chunk
            remaining_tokens = len(tokens) - start
            if remaining_tokens < min_chunk_size and len(chunks) > 0:
                # Adjust the start index to include more tokens from the previous chunk
                new_start = max(0, len(tokens) - chunk_size)
                chunk_tokens = tokens[new_start:len(tokens)]
                chunk_text =  model.tokenizer.convert_tokens_to_string(chunk_tokens)
                chunks[-1] = chunk_text  # Replace the last chunk with the adjusted chunk
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
            chunked_dialogues.append({'id': dialogue_id, 'sub_id': i, 'dialogue': chunk})
    return pd.DataFrame(chunked_dialogues)



def train(dialogues_path:Path, summary_pieces_path:Path):
    dialogues = pd.read_csv(dialogues_path)
    summary_pieces = pd.read_csv(summary_pieces_path)

    # Load the model
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    # Split the dialogues into chunks if they are too long
    chunked_dialogues = chunk_dialog(dialogues, model)
    

    pass



def main(dialog_path:str, summary_pieces_path:str, output_path:str):
    dialog_path = Path(dialog_path)
    summary_path = Path(summary_pieces_path)
    output_path = Path(output_path)

    # Check if the input files exist
    if not dialog_path.exists():
        raise FileNotFoundError(f"File not found: {dialog_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"File not found: {summary_path}")
    
    # Load dialogues and summary pieces
    dialogues = pd.read_csv(dialog_path)
    summary_pieces = pd.read_csv(summary_path)

    output_df = match_summary_pieces(dialogues, summary_pieces)
    output_df.to_csv(output_path, index=False)


if __name__ == '__main__':
    train_dialogues_path, train_summary_pieces_path = Path('reference_dialogues.csv'), Path('reference_summaries.csv')
    train(train_dialogues_path, train_summary_pieces_path)

    # test_dialogues_path, test_summary_pieces_path = 'dialogues.csv', 'summary_pieces.csv'
    # output_path = 'attributed_summary_pieces.csv'
    # main(test_dialogues_path, test_summary_pieces_path, output_path)