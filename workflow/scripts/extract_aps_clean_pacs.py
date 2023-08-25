import pandas as pd

PACS_PATH = snakemake.input[0]
pacs = pd.read_csv(PACS_PATH, index_col="DOI").sort_index()

for col in pacs.columns:
    pacs[col] = pacs[col].str.strip()  # Strip any extraneous tabs/spaces

    # Break any PACS codes that may have been concatenated together with no separator
    mask = pacs[col].str.len() == 16
    pacs.loc[mask, col] = (
        pacs.loc[mask, col].fillna("").str[0:8]
        + ","
        + pacs.loc[mask, col].fillna("").str[8:]
    )

    # Replace any odd separators
    pacs[col + "_clean"] = (
        pacs[col]
        .fillna("")
        .str.strip()
        .str.replace("1977", "")
        .str.replace(";", ",")
        .str.replace(" ", ",")
        .str.replace("and", ",")
        .str.split(":")
        .str[-1]
        .str.replace(".", "")
    )

# Join all PACS codes together in one column as a list
pacs["PACS_CODE"] = (
    pacs[["PACS1_clean", "PACS2_clean", "PACS3_clean", "PACS4_clean", "PACS5_clean"]]
    .agg(",".join, axis=1)
    .str.strip(",")
    .str.split(",")
)

# Drop previous columns
pacs = pacs.drop(
    labels=[
        "PACS1",
        "PACS2",
        "PACS3",
        "PACS4",
        "PACS5",
        "PACS1_clean",
        "PACS2_clean",
        "PACS3_clean",
        "PACS4_clean",
        "PACS5_clean",
    ],
    axis=1,
)

# Explode rows, strip spaces
pacs = pacs.explode("PACS_CODE")
pacs["PACS_CODE"] = pacs["PACS_CODE"].str.strip()

# Clean any remaining "." and weird separators
pacs["PACS_CODE"] = pacs["PACS_CODE"].str.replace(".", "")
pacs.loc[pacs.PACS_CODE.str[0] == "n", "PACS_CODE"] = pacs.loc[
    pacs.PACS_CODE.str[0] == "n", "PACS_CODE"
].str[1:]

# Drop everything that still isn't clean
pacs = pacs[
    (pacs.PACS_CODE.str.len() == 6)
    & (pacs.PACS_CODE.str[0].isin([str(i) for i in range(10)]))
]

# Get PACS codes per level
pacs["PACS_CODE_1"] = pacs.PACS_CODE.str[0]
pacs["PACS_CODE_2"] = pacs.PACS_CODE.str[0:2]
pacs["PACS_CODE_3"] = pacs.PACS_CODE.str[0:4]
pacs["PACS_CODE_5"] = pacs.PACS_CODE.str[0:6]

# Save
CLEAN_PACS_PATH = snakemake.output[0]
pacs.to_csv(CLEAN_PACS_PATH, header=True, index=True)
