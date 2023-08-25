# %%
# %load_ext autoreload
# %autoreload 2

import logging
import os
import sys

import emlens
import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc

sys.path.insert(0, "..")
import graph_embeddings
import residual_node2vec as rv

net = sparse.load_npz("net.npz")
node_table = pd.read_csv("node.csv", sep="\t")

community = node_table["community"]
is_hub_nodes = node_table["is_hub_nodes"]

net = net + net.T
# %%
models = {}

window_length = 10
models["LaplacianEigenMap"] = graph_embeddings.LaplacianEigenMap()
models["Node2Vec"] = rv.Node2Vec(window_length = window_length)
models["Residual2Vec"] = rv.Residual2Vec(window_length = window_length)
models["DeepWalk"] = rv.DeepWalk(window_length = window_length)

# %%
#
# Embeddings
#
embs = {}
for name, model in models.items():
    embs[name] = model.fit(net).transform(128)
    models[name] = model

# %%
#
# Projection to 2D
#
xys = {}
for name, emb in embs.items():
    xys[name] = PCA(n_components=2).fit_transform(emb)

dflist = []
for i, (name, xy) in enumerate(xys.items()):
    df = pd.DataFrame({"group": community})
    df["deg"] = np.array(net.sum(axis=1)).reshape(-1)
    df["x"] = xy[:, 0]
    df["y"] = xy[:, 1]
    df["model"] = name
    dflist += [df]
df = pd.concat(dflist)

g = sns.FacetGrid(
    data=df, col="model", hue="group", col_wrap=2, height=6, sharex=False, sharey=False
)

for i, (model, dg) in enumerate(df.groupby("model")):
    ax = g.axes.flat[i]
    sns.scatterplot(
        data=dg,
        x="x",
        y="y",
        size="deg",
        sizes=(1, 200),
        hue="group",
        edgecolor="black",
        linewidth=0.2,
        ax=ax,
    )
    ax.set_title(model)
    if i == 0:
        ax.legend(frameon=False)
    else:
        ax.legend().remove()

g.set_xlabels("")
g.set_ylabels("")
sns.despine()
g.axes[0].legend(frameon=False)

# %%
#
# How well does the embedding capture community structure?
#

# calc stat
results = []
deg = np.array(net.sum(axis=0)).reshape(-1)
for name, emb in embs.items():
    emb = np.asarray(emb, order="C")
    for k in [3, 5, 10, 15, 20, 25, 30]:
        q = emlens.f1_score(emb, node_table.community, k=k)
        results += [{"method": name, "q": q, "k":k}]
results = pd.DataFrame(results)


# %%
# visualize
sns.set(font_scale=1)
sns.set_style("white")
fig, axes = plt.subplots(figsize=(8, 4))
ax = sns.lineplot(data=results, x="k", y="q", hue = "method", ax=axes, marker = "o", markersize = 10)
ax.legend(frameon = False)
ax.set_ylabel("F1-score", fontsize=22)
sns.despine()
# %%
