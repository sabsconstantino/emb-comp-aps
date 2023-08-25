import sys

import numpy as np
import pandas as pd
import torch
import utils
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer

APS_TEXT_PATH = snakemake.input[0]
SCIBERT_EMBEDDING_PATH = snakemake.output[0]

# Load data
aps = pd.read_csv(APS_TEXT_PATH, index_col="DOI")

TEXT_COL = SCIBERT_EMBEDDING_PATH.split("_")[3].upper()

aps = aps[~((pd.isna(aps[TEXT_COL])) | (aps[TEXT_COL] == ""))]  # remove blank entries

# SciBERT embed
scibert_tokenizer = AutoTokenizer.from_pretrained(
    "allenai/scibert_scivocab_uncased", cache_dir="scibert_cache"
)
scibert_model = BertModel.from_pretrained(
    "allenai/scibert_scivocab_uncased",
    cache_dir="scibert_cache",
    output_hidden_states=True,
)
scibert_model = scibert_model.to("cuda:0")


def get_scibert_embedding(text):
    if isinstance(text, str):
        text = [text]
    # Encode the text, adding the (required!) special tokens, and converting to
    # PyTorch tensors.
    input_id = scibert_tokenizer(text, return_tensors="pt", add_special_tokens=True, padding=True)[
        "input_ids"
    ][:, :512]

    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers.
    with torch.no_grad():
        outputs = scibert_model(input_id.to("cuda:0"))
        hidden_states = outputs[2]  # Get the hidden states
        token_embeddings = torch.stack(
            hidden_states, dim=0
        )  # Concatenate all hidden layers into a big tensor
        token_embeddings = torch.squeeze(
            token_embeddings, dim=1
        )  # Remove the batch dimension
        if len(text) > 1:
            token_embeddings = token_embeddings.permute(1, 0, 2, 3)  # Permut
            last_layers = token_embeddings[:, -5:-1, :]
            embs = last_layers[:, :, 0, :].sum(axis = 1).cpu().numpy()
            #embs = last_layers.sum(axis=1).mean(axis=1).cpu().numpy()
        else:
            last_layers = token_embeddings[-5:-1, :]
            embs = last_layers[:, 0, :].sum(axis = 0).cpu().numpy().reshape((1, -1))
            #embs = last_layers.sum(axis=0).mean(axis=0).cpu().numpy().reshape((1, -1))
    return input_id.numpy(), embs


nchunks = np.ceil(aps.shape[0] / 200)  # split data to fit into GPU memory
prog_bar = tqdm(total=nchunks)  # show progress

emb_chunks = []
doi_chunks = []
for chunk in np.array_split(aps, nchunks):
    emb_chunks.append(get_scibert_embedding(chunk[TEXT_COL].values.tolist())[1])
    doi_chunks.append(chunk.index.values)
    prog_bar.update()

emb = np.vstack(emb_chunks)
doi = np.concatenate(doi_chunks)
np.savez_compressed(SCIBERT_EMBEDDING_PATH, emb=emb, doi=doi)
