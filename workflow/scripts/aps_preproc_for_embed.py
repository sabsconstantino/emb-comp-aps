import pandas as pd

APS_RAW_DATA = snakemake.input[0]
APS_METADATA = snakemake.input[1]
APS_JOURNAL_SETS = snakemake.params.journal_set
APS_MAX_YEAR = snakemake.params.max_year

APS_PROCESSED_DIR = snakemake.output[0]

df = pd.read_csv(APS_RAW_DATA, header=0)
df = df.rename(columns={"citing_doi": "CITING_DOI", "cited_doi": "CITED_DOI"})
df = df.assign(CITING_JOURNAL=lambda x: x.CITING_DOI.str[8:].str.split(".").str[0])
df = df.assign(CITED_JOURNAL=lambda x: x.CITED_DOI.str[8:].str.split(".").str[0])

df_meta = pd.read_csv(APS_METADATA, header=0, index_col="DOI").drop("META_FILENAME", axis=1)
df_meta = df_meta.assign(DATE=df_meta.DATE.apply(lambda x: pd.to_datetime(x)))
df_meta["YEAR"] = df_meta.DATE.dt.year

# Filter to remove non-peer-reviewed article entries
TO_REMOVE = ["erratum", "comment", "reply", "editorial", "essay", "retraction", "announcement"]
df_meta = df_meta[~df_meta.ARTICLE_TYPE.isin(TO_REMOVE)]

for journal_set in APS_JOURNAL_SETS:
    # Filter for APS journals of interest
    if "all" in journal_set:  # don't filter, use all journals
        df_filtered = df
    else:
        df_filtered = df[
            (df.CITED_JOURNAL.isin(journal_set)) & (df.CITING_JOURNAL.isin(journal_set))
        ]

    # Filter for years of interest
    df_filtered = df_filtered.merge(
        df_meta[["DATE", "YEAR"]], left_on="CITING_DOI", right_index=True, how="inner",
    ).rename(columns={"DATE": "CITING_DATE", "YEAR": "CITING_YEAR"})
    df_filtered = df_filtered.merge(
        df_meta[["DATE", "YEAR"]], left_on="CITED_DOI", right_index=True, how="inner",
    ).rename(columns={"DATE": "CITED_DATE", "YEAR": "CITED_YEAR"})
    df_filtered = df_filtered[(df_filtered.CITING_YEAR >= df_filtered.CITED_YEAR) & (df_filtered.CITING_YEAR <= APS_MAX_YEAR)]

    df_filtered.to_csv(APS_PROCESSED_DIR, index=False)
    print("Saved citation network to " + str(APS_PROCESSED_DIR))
