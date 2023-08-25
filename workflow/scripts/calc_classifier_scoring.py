import pandas as pd
from sklearn.metrics import (classification_report, f1_score, precision_score,
                             recall_score)

# Define paths
KNN_PATH_TXT = snakemake.input[0]
CIT_CLASS_PATH = snakemake.input[1]
SCORE_PATH = snakemake.output[0]

# Extract paths of kNN results
with open(KNN_PATH_TXT, "r") as f:
    KNN_PATHS = f.read()
    KNN_PATHS = KNN_PATHS.strip()
    KNN_PATHS = KNN_PATHS.split(" ")

# Initiate output dataframe
df = pd.DataFrame(
    columns=[
        "EMBEDDING",
        "IS_ALL_JOURNALS",
        "K",
        "AVERAGE",
        "TRIAL_NUM",
        "PRECISION",
        "RECALL",
        "F1_SCORE",
    ]
)
df = df.set_index(["EMBEDDING", "IS_ALL_JOURNALS", "K", "AVERAGE", "TRIAL_NUM"])

# Average calculations for multiclass precision/recall/F1
AVGS = ["micro", "macro", "weighted"]

# Classes
CLASSIF = SCORE_PATH.split("_")[4]
CLASSIF = CLASSIF.upper() if CLASSIF == "pacs" else CLASSIF

# Fill dataframe with precision, recall, F1 scores of KNN classifications
for filename in KNN_PATHS:
    params = filename.split("_")
    params[-1] = params[-1].replace(".csv", "")  # remove file extension
    emb_method = params[4]  # get embedding method
    params = params[5:]  # keep only kNN related params
    params = dict(
        [p.split("-") for p in params]
    )  # change params to dictionary with format param: value
    is_all_journals = int(params["alljournals"])
    k = int(params["k"])
    trial = int(params["trial"])
    with open(filename, "r") as f:
        knn = pd.read_csv(f, sep=",")
        for avg in AVGS:
            df.loc[
                (emb_method, is_all_journals, k, avg, trial), "PRECISION"
            ] = precision_score(knn.TRUE_LABEL, knn.PRED_LABEL, average=avg)
            df.loc[
                (emb_method, is_all_journals, k, avg, trial), "RECALL"
            ] = recall_score(knn.TRUE_LABEL, knn.PRED_LABEL, average=avg)
            df.loc[(emb_method, is_all_journals, k, avg, trial), "F1_SCORE"] = f1_score(
                knn.TRUE_LABEL, knn.PRED_LABEL, average=avg
            )

# Load citation network classification, calculate metrics
cit = pd.read_csv(CIT_CLASS_PATH, index_col=["IS_ALL_JOURNALS", "DOI"])
CLASS_COL = (
    "PACS_1" if CLASSIF.casefold() == "pacs" else "JOURNAL"
)  # column to extract from citation classification
for avg in AVGS:
    cit_focus = cit.loc[0]
    cit_all = cit.loc[1]

    # Precision
    df.loc[("citation", 0, 0, avg, 0), "PRECISION"] = precision_score(
        cit_focus["TRUE_" + CLASS_COL].dropna().values,
        cit_focus["CLASS_" + CLASS_COL].dropna().values,
        average=avg,
    )  # focus journals only (i.e. PRA-E)
    df.loc[("citation", 1, 0, avg, 0), "PRECISION"] = precision_score(
        cit_all["TRUE_" + CLASS_COL].dropna().values,
        cit_all["CLASS_" + CLASS_COL].dropna().values,
        average=avg,
    )  # all journals

    # Recall
    df.loc[("citation", 0, 0, avg, 0), "RECALL"] = recall_score(
        cit_focus["TRUE_" + CLASS_COL].dropna().values,
        cit_focus["CLASS_" + CLASS_COL].dropna().values,
        average=avg,
    )  # focus journals only (i.e. PRA-E)
    df.loc[("citation", 1, 0, avg, 0), "RECALL"] = recall_score(
        cit_all["TRUE_" + CLASS_COL].dropna().values,
        cit_all["CLASS_" + CLASS_COL].dropna().values,
        average=avg,
    )  # all journals

    # F1 score
    df.loc[("citation", 0, 0, avg, 0), "F1_SCORE"] = f1_score(
        cit_focus["TRUE_" + CLASS_COL].dropna().values,
        cit_focus["CLASS_" + CLASS_COL].dropna().values,
        average=avg,
    )  # focus journals only (i.e. PRA-E)
    df.loc[("citation", 1, 0, avg, 0), "F1_SCORE"] = f1_score(
        cit_all["TRUE_" + CLASS_COL].dropna().values,
        cit_all["CLASS_" + CLASS_COL].dropna().values,
        average=avg,
    )  # all journals

# Average scores across trials
df["PRECISION"] = pd.to_numeric(df.PRECISION)
df["RECALL"] = pd.to_numeric(df.RECALL)
df["F1_SCORE"] = pd.to_numeric(df.F1_SCORE)
df = df.groupby(["EMBEDDING", "IS_ALL_JOURNALS", "K", "AVERAGE"]).mean()
df = (
    df.reset_index()
    .set_index(["EMBEDDING", "IS_ALL_JOURNALS", "AVERAGE", "K"])
    .sort_index()
)

# Save
df.to_csv(SCORE_PATH, index=True, header=True)
