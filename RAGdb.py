import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import fitz 
import pandas as pd
from spacy.lang.en import English 
from tqdm.auto import tqdm 

from sentence_transformers import SentenceTransformer

filepath = "localRAGlib"
files = os.listdir(filepath)
data_df_save_path = "localRAG.csv"
embeddings_df_save_path = "localRAG_embs.csv"
device = "mps:0"
emb_model_name = "mixedbread-ai/mxbai-embed-large-v1" 

""" 
all-mpnet-base-v2
"""

embedding_model = SentenceTransformer(model_name_or_path=emb_model_name, 
                                      device=device)

nlp = English()
nlp.add_pipe("sentencizer")
num_sentence_chunk_size = 20


def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    text = text.replace("\n", " ").replace("- ", "").strip() #newlines and line breaks
    text = text.replace(" . .", "")
    # Other potential text formatting functions can go here

    return text


def split_list(input_list: list, 
               slice_size: int) -> list[list[str]]:
    """Splits a list of chunked sentences into slice_size chunks"""
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]


def make_dict(text_chunk, page_items):
    """Util function for making text chunk db"""
    chunk_dict = {
        "sentence_chunk" : text_chunk,
        "document" : page_items["document"],
        "page_number": page_items["page_number"],
        "chunk_char_count" : len(text_chunk),
        "chunk_word_count" : len([word for word in text_chunk.split(" ")]),
        "chunk_token_count" : len(text_chunk) / 4,# 1 token = ~4 characters
    }
    return chunk_dict



pages_and_texts = []
for file in tqdm(files):
    pdf_path = os.path.join(filepath,file)
    doc = fitz.open(pdf_path)
    for page_number, page in enumerate(doc):
        text = page.get_text() 
        text = text_formatter(text)
        # Filter out mostly empty pages and weirdly formatted pages
        if len(text.split(" ")) >= 50 and len(text) <= 5000: 
            pages_and_texts.append({"document": file,
                                    "page_number": page_number, 
                                    "page_char_count": len(text),
                                    "page_word_count": len(text.split(" ")),
                                    "page_sentence_count_raw": len(text.split(". ")),
                                    "page_token_count": len(text) / 4,  
                                    "text": text})

print("Success loading in docs")

chunks_and_texts = []
for item in tqdm(pages_and_texts):
    doc = nlp(item['text'])
    sentence_chunks = list(doc.sents)
    text_chunks = split_list(sentence_chunks,num_sentence_chunk_size)
    for chunk in text_chunks:
        sentences = [" " + str(sentence) for sentence in chunk]
        joined_sentence_chunk =  "".join(sentences).replace("  ", " ").strip()
        num_chars = len(joined_sentence_chunk)
        #overly short chunks get ignored
        if num_chars > 100:
            chunk_dict = make_dict(joined_sentence_chunk, item)
            chunks_and_texts.append(chunk_dict)

df = pd.DataFrame(chunks_and_texts)
text_chunks = df["sentence_chunk"]

print("Success chunking docs")

text_chunk_embeddings = embedding_model.encode(text_chunks,
                                               batch_size=32,
                                               convert_to_tensor=True)

print("Success embedding chunks")

text_chunk_embeddings = text_chunk_embeddings.to('cpu').numpy()
embeddings_df = pd.DataFrame(text_chunk_embeddings)

df.to_csv(data_df_save_path, index=False)

embeddings_df.to_csv(embeddings_df_save_path, index=False)

print("Success creating CSV")

