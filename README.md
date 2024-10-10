# Conversation Summary Attribution and Ordering

## Project Overview
This project aims to solve the problem of attributing summary pieces to their corresponding conversations and ordering these summaries according to the chronological flow of each dialog. The solution involves using a pre-trained model, `sentence-transformers/all-mpnet-base-v2`, which has been fine-tuned to capture the similarities between the summary pieces and conversation parts.

## Goals
1. **Attribute Summary Pieces**: Identify which parts of the conversation each summary piece corresponds to.
2. **Chronological Ordering**: Ensure that the attributed summary pieces align with the order of the conversation.

## Approach
1. **Model Fine-Tuning**: 
   - Utilized the `sentence-transformers/all-mpnet-base-v2` model as a base.
   - Fine-tuned the model to better capture similarities between summary pieces and corresponding conversation parts.
2. **Splitting Conversations**:
   - Split conversations into smaller chunks to create more concise vector representations, improving alignment accuracy.
3. **Ordering Summaries**:
   - Checked the alignment of summary pieces with conversation chunks.
   - If overlaps occurred, further split the conversation into sentences to achieve more accurate ordering.

## Results
The approach successfully attributed each summary piece to the correct conversation chunk and maintained the chronological order of summaries, providing a coherent summary of the dialogs.