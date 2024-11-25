Having fun making a couple textbooks into a RAG. 

localRAG.ipynb localMistral.ipynb are notebooks used to mess around and get the code working. 

RAGdb.py will read in PDFs contained in a subdirectory localRAGlib and initialize 2 csv files: localRAG.csv that contains chunked text + metadata, and localRAG_embeds.csv that contains the data of a vector database for chunk retrieval. The model used for this is mixedbread-ai/mxbai-embed-large-v1. TODO: better text extraction from various PDFs (especially symbols/equations/graphs/tables); ideally things should be extracted into markdown/TeX.

RAGplay.ipynb has code to run queries using this data; currently the code is written to use google/gemma-2-2b-it. 
