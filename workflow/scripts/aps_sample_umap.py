from itertools import product
from os.path import join
import pandas as pd
import networkx as nx
from scipy import sparse
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import math
from bs4 import BeautifulSoup

EMBEDDING_PATH = snakemake.input[0]
META_PATH = snakemake.input[1]
PACS_PATH = snakemake.input[2]
CIT_NET_PATH = snakemake.input[3]

UMAP_PATH = snakemake.output[0]

# Load embedding vector
emb = pd.read_csv(EMBEDDING_PATH, header=0, index_col="DOI").sort_index()
DIM = emb.shape[1]
emb.columns = list(range(0,DIM))

# Load metadata for titles, preprocess titles for better readability
meta = pd.read_csv(META_PATH, index_col="DOI").drop(columns="META_FILENAME")
meta = meta[~((pd.isna(meta.TITLE)) | (meta.TITLE == ""))]
meta["TITLE"] = meta["TITLE"].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text()) # parse any html
meta["TITLE"] = meta["TITLE"].str.replace("/emph>","") # removes any "straggling" or extra html tags that BeautifulSoup did not catch for some reason
meta["TITLE"] = meta["TITLE"].str.strip() # remove extra spaces

# Load PACS, dedupe
pacs = pd.read_csv(PACS_PATH, header=0, dtype="str")
pacs = (
    pacs.groupby(["DOI", "PACS_CODE_1"])
        .count()
        .reset_index()
        .sort_values(by=["PACS_CODE", "PACS_CODE_1"])
        .drop_duplicates(subset="DOI", keep="last")[["DOI", "PACS_CODE_1"]]
        .set_index("DOI")
        .sort_index()
)

# Load citation net, get citation counts
cit = pd.read_csv(CIT_NET_PATH)
cit_ct = cit.groupby("CITED_DOI").CITING_DOI.nunique()
cit_ct = cit_ct.to_frame().rename(columns={"CITING_DOI":"CIT_CT"})
cit_ct.index = cit_ct.index.rename("DOI")

# Generate UMAP
reducer = umap.UMAP()
samp = emb.sample(frac=0.2)
sample_doi = samp.index
umap_embedding = reducer.fit_transform(samp.values)
df = pd.DataFrame(umap_embedding)
df["DOI"] = sample_doi
df = df.set_index("DOI").merge(pacs, left_index=True, right_index=True)
df = df.merge(meta.TITLE, left_index=True, right_index=True)
df = df.merge(cit_ct, left_index=True, right_index=True, how="left")
df["CIT_CT"] = df["CIT_CT"].fillna(0)
df = df.rename(columns={0:"x", 1:"y"})

# Save
df.to_csv(UMAP_PATH, header=True, index=True)
