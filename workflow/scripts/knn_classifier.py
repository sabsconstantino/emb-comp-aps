import pandas as pd
from sklearn.model_selection import train_test_split

# from sklearn.neighbors import KNeighborsClassifier
import faiss
import numpy as np
from scipy import stats
from sklearn.preprocessing import normalize

# Find the nearest neighbors using faiss
def faiss_knn(emb, y_train, k):
    gpu_resource = faiss.StandardGpuResources()  # initiate gpu resource 
    cpu_index = faiss.IndexFlatL2(emb.shape[1])  # create cpu index
    gpu_index = faiss.index_cpu_to_gpu(gpu_resource, 0, cpu_index)  # transfer cpu index to gpu
    gpu_index.add(np.ascontiguousarray(emb.astype(np.float32)))

    def predict(target):
        distances, indices = gpu_index.search(target.astype(np.float32), k=k)
        ypred = y_train[indices]
        return np.array(stats.mode(ypred, axis=1)[0]).reshape(-1)

    return predict


# Data
DATA_PATH_PACS_CLEAN = snakemake.input[0]
DATA_PATH_EMBEDDING = snakemake.input[1]
DATA_PATH_ABSTRACT = snakemake.input[2]

# Params
PARAM_FOCUS_JOURNALS = snakemake.params.focus_journals

# Output
OUTPUT_CLASSIFICATION = snakemake.output[0]

# Extract other params from output filename
# - we assume that kNN params are in the last 5 parts of output filename:
#   1. alljournals (0 or 1)
#   2. withabstracts (0 or 1) -- if 1, we only consider all papers with abstracts
#   3. classes (journal or PACS)
#   4. k (no. of neighbors)
#   5. trial (nth model/exp)
PARAM_KNN = OUTPUT_CLASSIFICATION.split("_")[-5:]
PARAM_KNN[-1] = PARAM_KNN[-1].replace(".csv","")  # ensure ".csv" extension is removed
PARAM_KNN = dict([p.split("-") for p in PARAM_KNN])

# Load embedding
emb = pd.read_csv(DATA_PATH_EMBEDDING, header=0, index_col="DOI").sort_index()
DIM = emb.shape[1]
emb.columns = list(range(0, DIM))
emb["JOURNAL"] = emb.index.str[8:].str.split(".").str[0]

# Filter vectors if needed
if not int(PARAM_KNN["alljournals"]):
    emb = emb[emb.JOURNAL.isin(PARAM_FOCUS_JOURNALS)]
if int(PARAM_KNN["withabstracts"]):
    abstracts = pd.read_csv(DATA_PATH_ABSTRACT, index_col="DOI")
    emb = emb.merge(abstracts, left_index=True, right_index=True)
    emb = emb.drop(labels="ABSTRACT", axis=1)

# We use this variable for the column name of the classes we will predict
CLS = "JOURNAL"  # by default, predict journal

if PARAM_KNN["classes"].casefold() == "pacs":  # predict PACS (level 1 only)
    # Load PACS codes
    pacs = pd.read_csv(DATA_PATH_PACS_CLEAN, header=0, dtype="str")
    # Dedupe PACS, keep most frequent
    pacs = (
        pacs.groupby(["DOI", "PACS_CODE_1"])
        .count()
        .reset_index()
        .sort_values(by=["PACS_CODE", "PACS_CODE_1"])
        .drop_duplicates(subset="DOI", keep="last")[["DOI", "PACS_CODE_1"]]
        .set_index("DOI")
        .sort_index()
    )
    emb = emb.merge(pacs, left_index=True, right_index=True)
    CLS = "PACS_CODE_1"
print("Number of embedding dimensions: {}".format(DIM))
print("Working dataframe shape: {}".format(emb.shape))

# Split data, normalize
X_train, X_test, y_train, y_test = train_test_split(
    emb[range(0, DIM)].values, emb[CLS], test_size=0.2
)
X_train = normalize(X_train)
X_test = normalize(X_test)

# Train classifier
# nn = KNeighborsClassifier(n_neighbors=int(PARAM_KNN["k"]), n_jobs=32)
# nn.fit(X_train, y_train)
predict = faiss_knn(X_train, y_train, k=int(PARAM_KNN["k"]))

# Predict
# y_pred = nn.predict(X_test)
y_pred = predict(X_test)

# Save
df = pd.DataFrame(list(zip(y_test, y_pred)), columns=["TRUE_LABEL", "PRED_LABEL"])
df.index = y_test.index
df["TRIAL_NUM"] = PARAM_KNN["trial"]
df["IS_ALL_JOURNALS"] = int(PARAM_KNN["alljournals"])
df["K"] = int(PARAM_KNN["k"])
df.to_csv(OUTPUT_CLASSIFICATION, index=True)
