import json

import pandas as pd

RAW_CITATION_PATH = snakemake.input[0]
df = pd.read_csv(RAW_CITATION_PATH, header=0)
df = df.rename(columns={"citing_doi": "CITING_DOI", "cited_doi": "CITED_DOI"})
df = df.assign(CITING_JOURNAL=lambda x: x.CITING_DOI.str[8:].str.split(".").str[0])
df = df.assign(CITED_JOURNAL=lambda x: x.CITED_DOI.str[8:].str.split(".").str[0])

RAW_METADATA_PATH = snakemake.input[1]
RAW_METADATA_PATH_JOURNAL = {
    "PhysRev": "PR",
    "PhysRevA": "PRA",
    "PhysRevAccelBeams": "PRAB",
    "PhysRevApplied": "PRAPPLIED",
    "PhysRevB": "PRB",
    "PhysRevC": "PRC",
    "PhysRevD": "PRD",
    "PhysRevE": "PRE",
    "PhysRevFluids": "PRFLUIDS",
    "PhysRevSeriesI": "PRI",
    "PhysRevLett": "PRL",
    "PhysRevMaterials": "PRMATERIALS",
    "PhysRevPhysEducRes": "PRPER",
    "PhysRevSTAB": "PRSTAB",
    "PhysRevSTPER": "PRSTPER",
    "PhysRevX": "PRX",
    "RevModPhys": "RMP",
    "Physics": "",
    "PhysicsPhysiqueFizika": "",
    "PhysRevFocus": "",
}


def extract_json_filename(x):
    paper = x.DOI.split("/")[1]
    journal = paper.split(".")[0]
    paper_volume = paper.split(".")[1]
    return (
        RAW_METADATA_PATH
        + RAW_METADATA_PATH_JOURNAL[journal]
        + "/"
        + paper_volume
        + "/"
        + paper
        + ".json"
    )


def extract_metadata(x):
    cleaned_data = {"DOI": x.DOI, "DATE": None, "ARTICLE_TYPE": None, "TITLE": None, "JOURNAL_ID": None, "JOURNAL_NAME": None, "JOURNAL_ABBREV": None}
    try:
        with open(x.META_FILENAME) as json_data:
            data = json.load(json_data)
            cleaned_data["DATE"] = data.get("date", "")
            cleaned_data["ARTICLE_TYPE"] = data.get("articleType", "")
            cleaned_data["TITLE"] = data.get("title", "").get("value", "")
            cleaned_data["JOURNAL_ID"] = data.get("journal", "").get("id", "")
            cleaned_data["JOURNAL_NAME"] = data.get("journal", "").get("name", "")
            cleaned_data["JOURNAL_ABBREV"] = data.get("journal", "").get("abbreviatedName", "")
    except FileNotFoundError:
        pass
    return cleaned_data


EXTR_METADATA_PATH = snakemake.output[0]
meta = pd.DataFrame(df["CITING_DOI"].append(df["CITED_DOI"]).unique(), columns=["DOI"])
meta = meta.assign(META_FILENAME=meta.apply(extract_json_filename, axis=1))
meta = meta.merge(
    pd.DataFrame(list(meta.apply(extract_metadata, axis=1))), on="DOI", how="left"
)
meta.to_csv(EXTR_METADATA_PATH, header=True, index=False)
