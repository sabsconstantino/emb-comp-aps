import pandas as pd
from bs4 import BeautifulSoup

# Load data
APS_METADATA_PATH = snakemake.input[0]
RAW_WOS_APS_ABSTRACT = snakemake.input[1]
aps = pd.read_csv(APS_METADATA_PATH, index_col="DOI", parse_dates=["DATE"])
aps = aps.drop(columns="META_FILENAME")
abstracts = pd.read_csv(APS_ABS_PATH, header=0, usecols=["DOI", "ABSTRACT"], index_col="DOI")
aps = aps.merge(abstracts, left_index=True, right_index=True, how="left")

APS_TEXT_PATH = snakemake.output[0]
TEXT_COL = APS_TEXT_PATH.split("_")[2].upper()[:-3]  # get text property from filename, then remove extension

aps_text = aps[[TEXT_COL]].copy()
aps_text = aps_text[~((pd.isna(aps_text[TEXT_COL])) | (aps_text[TEXT_COL] == ""))]  # remove blank entries
aps_text[TEXT_COL] = aps_text[TEXT_COL].apply(
    lambda x: BeautifulSoup(x, "html.parser").get_text()
)  # parse any html
aps_text[TEXT_COL] = aps_text[TEXT_COL].str.replace(
    "/emph>", ""
)  # removes any "straggling" or extra html tags that BeautifulSoup did not catch for some reason
aps_text[TEXT_COL] = aps_text[TEXT_COL].str.strip()  # remove extra spaces

aps_text.to_csv(APS_TEXT_PATH, header=True, index=True)
