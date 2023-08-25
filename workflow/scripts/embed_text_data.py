from gensim.models import doc2vec
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

APS_METADATA_PATH = snakemake.input[0]
SCIBERT_EMB_PATH = snakemake.input[1]
EXTR_APS_TEXT_PATH = snakemake.input[2]
APS_EMBEDDING_PATH = snakemake.output[0]

# Parameters
TO_REMOVE = snakemake.params.to_remove
JOURNAL_SET = APS_EMBEDDING_PATH.split("_")[1].split("-")
MAX_YEAR = APS_EMBEDDING_PATH.split("_")[2]
TEXT_CONTENT = APS_EMBEDDING_PATH.split("_")[3]
EMB_METHOD = APS_EMBEDDING_PATH.split("_")[4]

# Load metadata
aps = pd.read_csv(APS_METADATA_PATH, index_col="DOI", parse_dates=["DATE"])
aps["JOURNAL_ABBREV"] = aps.index.str[8:].str.split(".").str[0]
if "all" in JOURNAL_SET:
    pass
else:
    aps = aps[aps.JOURNAL_ABBREV.isin(JOURNAL_SET)]
aps = aps[aps.DATE <= "{}-12-31".format(MAX_YEAR)]
aps = aps[~aps.ARTICLE_TYPE.isin(TO_REMOVE)]

# Load processed text
txt = pd.DataFrame(EXTR_APS_TEXT_PATH, index_col="DOI")
txt = txt[txt.index.isin(aps.index)]

if EMB_METHOD.casefold() == "scibert":
    # Load SciBERT embedding of all text
    with np.load(SCIBERT_EMB_PATH, allow_pickle=False) as f:
        emb = f["emb"]
    with np.load(SCIBERT_EMB_PATH, allow_pickle=True) as f:
        doi = f["doi"]
    df_emb = pd.DataFrame(emb)
    df_emb["DOI"] = doi
    df_emb = df_emb[["DOI"] + list(range(0, emb.shape[1]))]  # rearrange cols
    df_emb = df_emb.set_index("DOI")
    df_emb = df_emb.merge(txt[TEXT_CONTENT], left_index=True, right_index=True).drop(columns="TITLE")
elif EMB_METHOD.casefold() == "sentencebert":
    # Embed
    model = SentenceTransformer('paraphrase-mpnet-base-v2', device='cuda:1')
    emb = model.encode(txt[TEXT_CONTENT])

    # Transform into DataFrame
    df_emb = pd.DataFrame(data=emb, index=txt.index)
elif EMB_METHOD.casefold() == "doc2vec":
    # Embed
    sentences = [doc2vec.TaggedDocument(doc, [i]) for i, doc in enumerate(txt[TEXT_CONTENT].values)]
    model = doc2vec.Doc2Vec(sentences, vector_size=128, window_size=5, workers=16)

    # Transform into DataFrame
    vectors = []
    for i in range(len(titles)):
        vectors.append(model.docvecs[i])
    vectors = np.vstack(vectors)
    df_emb = pd.DataFrame(index=txt.index, data=vectors, columns=range(128))

df_emb.to_csv(APS_EMBEDDING_PATH, header=True, index=True)
