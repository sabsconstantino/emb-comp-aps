from os.path import join as j


configfile: "workflow/config.yaml"


###############################################################################
# Data
###############################################################################
DATA_DIR = config["data_dir"]

# Raw data
RAW_DATA_DIR = j(DATA_DIR, "raw")
RAW_APS_DIR = j(RAW_DATA_DIR, "APS")
RAW_APS_DATA = j(RAW_APS_DIR, "aps-dataset-citations-2018.csv")
RAW_APS_METADATA = j(RAW_APS_DIR, ) #config["aps_metadata_dir"]
RAW_APS_PACS = j(RAW_APS_DIR, "PACS.csv")

RAW_WOS_DIR = j(RAW_DATA_DIR, 'WOS')
RAW_WOS_APS_CITNET = j(RAW_WOS_DIR, 'wos_to_aps_degree0_and_1_category.bz2')
RAW_WOS_APS_ABSTRACT = j(RAW_WOS_DIR, 'wos_aps_abstracts.bz2')

# Derived data
DERIVED_DATA_DIR = j(DATA_DIR, "derived")

CIT_DIR = j(DERIVED_DATA_DIR, "journal_citations")
CIT_APS_CITATIONS = j(CIT_DIR, "aps_{journal_set}_{max_year}_citations.csv")

EXTR_APS_METADATA = j(DERIVED_DATA_DIR, "aps_dataset_metadata.bz2")
EXTR_APS_PACS_CLEAN = j(DERIVED_DATA_DIR, "cleaned_pacs.csv")
EXTR_APS_TEXT = j(DERIVED_DATA_DIR, "text", "aps_dataset_{text_content}.gz")

DOI_DIR = j(DERIVED_DATA_DIR, "doi_sets")
DOI_SET = j(DOI_DIR, "aps_{journal_set}_{max_year}_doi.csv")

STATS_DIR = j(DERIVED_DATA_DIR, "stats")
STATS_APS_YEARLY_PAPER_CNT = j(STATS_DIR, "aps_yearly_paper_count.csv")

CENTROID_DIST_DIR = j(DERIVED_DATA_DIR, 'centroid_dist')
CENTROID_DIST = j(CENTROID_DIST_DIR, 'centroid_dist_{journal_set}_{max_year}_{property_emb_method}.csv')

# Sampled paper pairs to be used when calculating embedding distance
APS_PAPER_PAIRS_DIR = j(DERIVED_DATA_DIR, "aps_paper_pairs")
APS_PAPER_PAIRS = j(APS_PAPER_PAIRS_DIR, "aps_{journal_set}_{max_year}_paper_pairs.gz")

###############################################################################
# Embedding
###############################################################################
# Embedding data
EMBEDDING_DATA_DIR = j(DATA_DIR, "embedding")
EMBEDDING_APS_TEXT_SCIBERT_ALL = j(
    EMBEDDING_DATA_DIR, "aps_all_2018_{text_content}_scibert_pretrain.npz"
) # pretrained scibert

# Embeddings to be trained
EMBEDDING_MODEL_DIR = j(EMBEDDING_DATA_DIR, "models")
EMBEDDING_VECTOR_DIR = j(EMBEDDING_DATA_DIR, "vectors")
EMBEDDING_APS_GRAPH_MODEL = j(
    EMBEDDING_MODEL_DIR, "aps_{journal_set}_{max_year}_{direction}_{graph_emb_method}_model"
)
EMBEDDING_APS_GRAPH_VECTOR = j(
    EMBEDDING_VECTOR_DIR,
    "aps_{journal_set}_{max_year}_{direction}_{graph_emb_method}_vector.gz",
)
EMBEDDING_APS_TEXT_VECTOR = j(
    EMBEDDING_VECTOR_DIR,
    "aps_{journal_set}_{max_year}_{text_content}_{text_emb_method}_vector.gz"
)
EMBEDDING_APS_GENERAL_VECTOR = j(
    EMBEDDING_VECTOR_DIR,
    "aps_{journal_set}_{max_year}_{property_emb_method}_vector.gz"
)

###############################################################################
# Classification
###############################################################################
CLASS_DIR = j(DATA_DIR, "classification")
CLASS_RESULT = j(
    CLASS_DIR,
    "aps_{journal_set}_{max_year}_{property_emb_method}_alljournals-{alljournals}_withabstracts-{withabstracts}_classes-{classes}_k-{k}_trial-{trial}.csv"
)
CLASS_RESULT_AGG = j(
    CLASS_DIR, 
    "aps_{journal_set}_{max_year}_{property_emb_method}_alljournals-{alljournals}_withabstracts-{{withabstracts}}_classes-{{classes}}_k-{k}_trial-{trial}.csv"
) # this refers to the same files as CLASS_RESULT above, but this wildcard is what we'll use when aggregating and scoring the results of the kNN classification
CLASS_RESULT_CIT_NET = j(
    CLASS_DIR,
    "aps_{journal_set}_{max_year}_cit_net_classes.csv"
)
CLASS_RESULT_EVAL_FILEPATHS = j(
    CLASS_DIR,
    "evaluation",
    "aps_{journal_set}_{max_year}_{property}_{withabstracts}_{classes}_knn_filepaths.txt"
)
CLASS_RESULT_EVAL_SCORE = j(
    CLASS_DIR,
    "evaluation",
    "aps_{journal_set}_{max_year}_{property}_{withabstracts}_{classes}_eval_scores.csv"
)

###############################################################################
# Wildcard Constraints and Supported Wildcards
###############################################################################
wildcard_constraints:
    journal_set = "[A-Za-z\-]+",
    max_year = "[0-9]+",
    direction = "undirected|tocited|tociting",
    text_content = "title|abstract|titleabstract|fulltext",
    property = "undirected|tocited|tociting|title|abstract|titleabstract",
    graph_emb_method = "residual2vec|node2vec|leigenmap",
    text_emb_method = "doc2vec|scibert|sentencebert",
    emb_method = "residual2vec|node2vec|leigenmap|doc2vec|scibert|sentencebert",
    classes = "journal|pacs|PACS",
    k = "[0-9]+",
    trial = "[0-9]+",

###############################################################################
# Wildcards
###############################################################################

JOURNAL_SETS = ["-".join(js) for js in config["journal_sets"]]
MAX_YEAR = config["max_year"]
GRAPH_DIRECTION = ["undirected"] #["undirected", "tocited"]
GRAPH_EMB_METHOD = ["leigenmap", "node2vec", "residual2vec"]
TEXT_CONTENT = ["title", "abstract"]
TEXT_EMB_METHOD = ["doc2vec", "scibert", "sentencebert"]
GRAPH_DIR_EMB_METHOD = expand("{direction}_{graph_emb_method}", direction=GRAPH_DIRECTION, graph_emb_method=GRAPH_EMB_METHOD)
TEXT_CONT_EMB_METHOD = expand("{content}_{text_emb_method}", content=TEXT_CONTENT, text_emb_method=TEXT_EMB_METHOD)
ALLJOURNALS = ["1"] # ["0", "1"]
WITHABSTRACTS = ["1"]
K = [str(2**i) for i in range(1,8)]
TRIAL = ["1","2","3"]
CLASSES = ["pacs"]  # ["pacs", "journal"]
TO_REMOVE = ["erratum", "comment", "reply", "editorial", "essay", "retraction", "announcement"]

# For use in classification result aggregation
def get_classif_wildcards(wildcards):
    if wildcards.property in GRAPH_DIRECTION:  # if classification is based on graph embedding
        prop_emb = expand("{direction}_{graph_emb_method}", direction=wildcards.property, graph_emb_method=GRAPH_EMB_METHOD)
    elif wildcards.property in TEXT_CONTENT:  # if classification is based on text embedding
        prop_emb = expand("{content}_{text_emb_method}", content=wildcards.property, text_emb_method=TEXT_EMB_METHOD)
    else:
        prop_emb = []
    return expand(CLASS_RESULT_AGG, journal_set=JOURNAL_SETS, max_year=MAX_YEAR, property_emb_method=prop_emb, alljournals=ALLJOURNALS, k=K, trial=TRIAL)

###############################################################################
# Rules
###############################################################################
rule all:
    input:
        expand(CLASS_RESULT, journal_set=JOURNAL_SETS, max_year=MAX_YEAR, property_emb_method=TEXT_CONT_EMB_METHOD+GRAPH_DIR_EMB_METHOD, alljournals=ALLJOURNALS, withabstracts=WITHABSTRACTS, k=K, trial=TRIAL, classes=CLASSES),
        expand(CLASS_RESULT_CIT_NET, journal_set=JOURNAL_SETS, max_year=MAX_YEAR),
        expand(APS_PAPER_PAIRS, journal_set=JOURNAL_SETS, max_year=MAX_YEAR),

rule aps_cit_net_classes:
    input:
        CIT_APS_CITATIONS,
        RAW_APS_PACS,
    params:
        focus_journals = ["PhysRevA","PhysRevB","PhysRevC","PhysRevD","PhysRevE"],
    output:
        CLASS_RESULT_CIT_NET,
    script:
        "workflow/scripts/cit_net_classifier.py"

rule aps_embedding_eval_knn:
    input:
        EXTR_APS_PACS_CLEAN,
        EMBEDDING_APS_GENERAL_VECTOR,
        RAW_WOS_APS_ABSTRACT,
    params:
        focus_journals = ["PhysRevA", "PhysRevB", "PhysRevC", "PhysRevD", "PhysRevE"],
    output:
        CLASS_RESULT,
    script:
        "workflow/scripts/knn_classifier.py"

rule aps_sample_paper_pairs:
    input:
        EXTR_APS_METADATA,
        EXTR_APS_PACS_CLEAN,
    params:
        to_remove = TO_REMOVE,
    output:
        APS_PAPER_PAIRS,
    script:
        "workflow/scripts/aps_sample_paper_pairs.py"

rule aps_text_embed:
    input:
        EXTR_APS_METADATA,
        EMBEDDING_APS_TEXT_SCIBERT_ALL,
        EXTR_APS_TEXT,
    params:
        to_remove = TO_REMOVE,
    output:
        EMBEDDING_APS_TEXT_VECTOR,
    script:
        "workflow/scripts/embed_text_data.py"

rule aps_embed_all_scibert:
    input:
        EXTR_APS_TEXT,
    output:
        EMBEDDING_APS_TEXT_SCIBERT_ALL,
    script:
        "workflow/scripts/embed_all_scibert.py"

rule aps_text_preproc:
    input:
        EXTR_APS_METADATA,
        RAW_WOS_APS_ABSTRACT,
    output:
        EXTR_APS_TEXT,
    script:
        "workflow/scripts/aps_preproc_text.py"

rule aps_citation_embed:
    input:
        CIT_APS_CITATIONS,
    output:
        EMBEDDING_APS_GRAPH_VECTOR,
    script:
        "workflow/scripts/embed_citation_network.py"

rule aps_citation_preproc:
    input:
        RAW_APS_DATA,
        EXTR_APS_METADATA,
    params:
        journal_set=config["journal_sets"],
        max_year=config["max_year"],
    output:
        CIT_APS_CITATIONS,
    script:
        "workflow/scripts/aps_preproc_for_embed.py"

rule aps_extract_clean_pacs:
    input:
        RAW_APS_PACS,
    output:
        EXTR_APS_PACS_CLEAN,
    script:
        "workflow/scripts/extract_aps_clean_pacs.py"

rule aps_extract_metadata:
    input:
        RAW_APS_DATA,
        RAW_APS_METADATA,
    output:
        EXTR_APS_METADATA,
    script:
        "workflow/scripts/extract_aps_metadata.py"
