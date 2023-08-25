import networkx as nx
import pandas as pd
import sys
sys.path.insert(0, "libs/graph_embeddings/")
import graph_embeddings
import residual_node2vec as r2v

import embedding

# input/output
DATA_PATH = snakemake.input[0]
EMBEDDING_PATH = snakemake.output[0]

# Params
DIMS = 128
NUM_WORKERS = 8
WALK_LENGTH = 80
NUM_WALKS = 10
WINDOW_SIZE = 10
RESTART_PROB = 0.01

# Retrieve data, initiate graph
df = pd.read_csv(DATA_PATH)
if "undirected" in str(EMBEDDING_PATH):  # if we want an undirected graph
    G = nx.Graph()
else:  # assume directed graph by default
    G = nx.DiGraph()

# Build graph
nodelist = df["CITING_DOI"].append(df["CITED_DOI"]).unique()
G.add_nodes_from(nodelist)
if "tociting" in str(
    EMBEDDING_PATH
):  # if we want a directed graph with edges from reference paper to citing paper
    G.add_edges_from(df[["CITED_DOI", "CITING_DOI"]].values)
else:  # by default, assume direction from citing paper towards cited paper/reference (though it doesn't matter if graph is undirected)
    G.add_edges_from(df[["CITING_DOI", "CITED_DOI"]].values)

# Embed
if "residual2vec" in str(EMBEDDING_PATH):
    model = r2v.Residual2Vec(
        null_model="configuration",
        window_length=WINDOW_SIZE,
        num_walks=NUM_WALKS,
        restart_prob=RESTART_PROB,
    )
    model = model.fit(G)
    vector = model.transform(DIMS)
    vector_df = pd.DataFrame(vector)
    vector_df["DOI"] = nodelist
    vector_df = vector_df[["DOI"] + list(range(0, DIMS))]
elif "leigenmap" in str(EMBEDDING_PATH):
    model = graph_embeddings.LaplacianEigenMap()
    model = model.fit(G)
    vector = model.transform(DIMS)
    vector_df = pd.DataFrame(vector)
    vector_df["DOI"] = nodelist
    vector_df = vector_df[["DOI"] + list(range(0, DIMS))]
else:  # if "node2vec"
    entity = {nodelist[i]: i for i in range(len(nodelist))}
    node2vec = embedding.Node2Vec(
        G,
        entity,
        dimensions=DIMS,
        walk_length=WALK_LENGTH,
        num_walks=NUM_WALKS,
        workers=NUM_WORKERS,
        window_size=WINDOW_SIZE,
    )
    node2vec.simulate_walks()
    vector = node2vec.learn_embedding()
    vector_df = pd.DataFrame(vector).T.reset_index().rename(columns={"index": "DOI"})
    # model = r2v.Node2Vec(num_walks=num_walks, walk_length=walk_length, window_length=window_size)
    # model = model.fit(G)
    # vector = model.transform(DIMS)
    # vector_df = pd.DataFrame(vector)
    # vector_df["DOI"] = nodelist
    # vector_df = vector_df[["DOI"] + list(range(0,DIMS))]

vector_df.to_csv(EMBEDDING_PATH, index=False)
