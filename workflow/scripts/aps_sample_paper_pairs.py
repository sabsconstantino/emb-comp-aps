import pandas as pd

APS_METADATA_PATH = snakemake.input[0]
PACS_PATH = snakemake.input[1]
PAIRS_PATH = snakemake.output[0]

PAIRS_FILENAME = PAIRS_PATH.split("/")[-1]
JOURNAL_SET = PAIRS_FILENAME.split("_")[1].split("-")
MAX_YEAR = PAIRS_FILENAME.split("_")[2]

meta = pd.read_csv(APS_METADATA_PATH, header=0, index_col="DOI", parse_dates=["DATE"])
pacs = pd.read_csv(
    PACS_PATH, header=0, usecols=["DOI", "PACS_CODE_1", "PACS_CODE_2"], dtype="str"
).drop_duplicates()
pacs = pacs.set_index("DOI")

MAX_DATE = pd.to_datetime("{}-12-31".format(MAX_YEAR))
meta = meta[meta.DATE <= MAX_DATE]  # filter date
meta["JOURNAL"] = meta.index.str[8:].str.split(".").str[0]
if "all" not in JOURNAL_SET:
    meta = meta[meta.JOURNAL.isin(JOURNAL_SET)]  # filter journal
pacs = pacs.merge(meta["JOURNAL"], left_index=True, right_index=True)

# Sample
df1 = pacs.sample(replace=True, frac=3)[["PACS_CODE_1", "PACS_CODE_2"]]
df2 = pacs.sample(replace=True, frac=3)[["PACS_CODE_1", "PACS_CODE_2"]]
pairs = pd.concat([df1.reset_index(), df2.reset_index()], axis=1)
pairs.columns = [
    "DOI_X",
    "PACS_CODE_1_X",
    "PACS_CODE_2_X",
    "DOI_Y",
    "PACS_CODE_1_Y",
    "PACS_CODE_2_Y",
]
pairs = pairs[pairs.DOI_X != pairs.DOI_Y]  # Ensure no same pairs
pairs = pairs.drop_duplicates()  # Drop duplicates

# Filter
same_pacs2 = pairs[pairs.PACS_CODE_2_X == pairs.PACS_CODE_2_Y].drop_duplicates(
    subset=["DOI_X", "DOI_Y"]
)  # same level 2 PACS (subdiscipline)
same_pacs1 = pairs[
    (pairs.PACS_CODE_1_X == pairs.PACS_CODE_1_Y)
    & (pairs.PACS_CODE_2_X != pairs.PACS_CODE_2_Y)
].drop_duplicates(
    subset=["DOI_X", "DOI_Y"]
)  # get pairs with same level 1 PACS but different level 2 PACS, deduplicate
same_pacs1 = pd.concat(
    [same_pacs1, same_pacs2]
)  # add back pairs with same level 2 PACS (since they have the same level 1 PACS too)
same_pacs1 = same_pacs1.drop_duplicates(
    subset=["DOI_X", "DOI_Y"], keep="last"
)  # deduplicate; if a pair has same PACS 2, keep that one
diff_pacs = pairs[(pairs.PACS_CODE_1_X != pairs.PACS_CODE_1_Y)].drop_duplicates(
    subset=["DOI_X", "DOI_Y"]
)  # get pairs with different PACS (discipline), deduplicate
diff_pacs = diff_pacs.merge(
    same_pacs1[["DOI_X", "DOI_Y"]], on=["DOI_X", "DOI_Y"], how="outer", indicator=True
)
diff_pacs = (
    diff_pacs[diff_pacs._merge == "left_only"].drop(columns="_merge").drop_duplicates()
)  # get only the papers with different PACS codes. do not count pairs of papers that also have mutual PACS codes
sample = pd.concat(
    [same_pacs1, diff_pacs]
)  # don't include same_pacs2 because those are also in same_pacs1

sample.to_csv(PAIRS_PATH, header=True, index=False)
