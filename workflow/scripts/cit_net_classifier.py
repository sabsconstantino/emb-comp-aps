import pandas as pd

CIT_NET_PATH = snakemake.input[0]
PACS_PATH = snakemake.input[1]
FOCUS_JOURNALS = snakemake.params.focus_journals

# Load preprocessed citation network
df = pd.read_csv(CIT_NET_PATH, header=0, dtype="str")

# Load and deduplicate PACS codes
# - dedupe by keeping the most frequently used level 1 PACS code (PACS_CODE_1)
pacs = pd.read_csv(PACS_PATH, header=0, index_col="DOI", dtype="str")
pacs = (
    pacs.groupby(["DOI", "PACS_CODE_1"])
    .count()
    .reset_index()
    .sort_values(by=["PACS_CODE", "PACS_CODE_1"])
    .drop_duplicates(subset="DOI", keep="last")[["DOI", "PACS_CODE_1"]]
    .set_index("DOI")
    .sort_index()
)

# Merge to get PACS codes of citation network
df = df.merge(
    pacs[["PACS_CODE_1"]], left_on="CITING_DOI", right_index=True, how="left"
).rename(columns={"PACS_CODE_1": "CITING_PACS_1"})
df = df.merge(
    pacs[["PACS_CODE_1"]], left_on="CITED_DOI", right_index=True, how="left"
).rename(columns={"PACS_CODE_1": "CITED_PACS_1"})

# In a separate dataframe, get citation network of papers in focus journals
df_focus = df[(df.CITING_JOURNAL.isin(FOCUS_JOURNALS)) & (df.CITED_JOURNAL.isin(FOCUS_JOURNALS))]

def classify_citation_net(cit):
    """Classify citation network into journals and level 1 PACS,
       based on majority rule of references.
    
       Params:
       cit -- citation network to be classified (pandas DataFrame)
    
       Returns:
       classified -- classified citation network
    """
    # Classify journals
    # - according to majority vote of cited (reference) journals
    # - break ties by alphabetic sorting
    journal_cls = (
        cit.groupby(["CITING_DOI", "CITING_JOURNAL", "CITED_JOURNAL"])
        .CITED_DOI.count()
        .to_frame()
        .reset_index()
        .sort_values(by=["CITED_DOI", "CITED_JOURNAL"], ascending=False)
    )
    journal_cls = (
        journal_cls.groupby(["CITING_DOI", "CITING_JOURNAL"])
        .head(1)
        .rename(
            columns={
                "CITED_JOURNAL": "CLASS_JOURNAL",
                "CITING_DOI": "DOI",
                "CITING_JOURNAL": "TRUE_JOURNAL",
            }
        )
        .set_index("DOI")
        .drop("CITED_DOI", axis=1)
    )
    # Classify PACS
    pacs_cls = (
        cit.groupby(["CITING_DOI", "CITING_PACS_1", "CITED_PACS_1"])
        .CITED_DOI.count()
        .to_frame()
        .reset_index()
        .sort_values(by=["CITED_DOI", "CITED_PACS_1"], ascending=False)
    )
    pacs_cls = (
        pacs_cls.groupby("CITING_DOI")
        .head(1)
        .rename(
            columns={
                "CITED_PACS_1": "CLASS_PACS_1",
                "CITING_PACS_1": "TRUE_PACS_1",
                "CITING_DOI": "DOI",
            }
        )
        .set_index("DOI")
        .drop("CITED_DOI", axis=1)
    )
    classified = journal_cls.merge(pacs_cls, left_index=True, right_index=True, how="left")
    return classified

# Merge, save
CLASSIF_PATH = snakemake.output[0]
classif_full = classify_citation_net(df)
classif_focus = classify_citation_net(df_focus)
classif = pd.concat([classif_focus, classif_full], keys=["0","1"], names=["IS_ALL_JOURNALS","DOI"])
classif.to_csv(CLASSIF_PATH, header=True, index=True)
