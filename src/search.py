import numpy as np
import cv2
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Paths
DESCRIPTORS_PATH = "features/descriptors/"
HISTOGRAMS_PATH = "features/histograms/"
MODELS_PATH = "features/models/"

def load_codebook():
    print("Loading codebook...")
    codebook = np.load(MODELS_PATH + "codebook.npy")
    with open(MODELS_PATH + "kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    return codebook, kmeans


def compute_query_histogram(image_path, kmeans):
    """ Extract SIFT → assign clusters → compute histogram for query """
    sift = cv2.SIFT_create()
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kp, des = sift.detectAndCompute(gray, None)
    if des is None:
        print("⚠ لا يوجد ميزات في الصورة!")
        return None

    labels = kmeans.predict(des)
    hist, _ = np.histogram(labels, bins=np.arange(kmeans.n_clusters + 1))
    hist = hist / hist.sum()

    return hist


def load_database_histograms():
    """ Load all histograms and keep file names """
    hists = []
    names = []

    for fname in os.listdir(HISTOGRAMS_PATH):
        if fname.endswith(".npy"):
            hists.append(np.load(HISTOGRAMS_PATH + fname))
            names.append(fname)

    return np.array(hists), names


def search(query_image, top_k=5):
    print("\n🔎 Starting image retrieval...\n")

    codebook, kmeans = load_codebook()
    db_hists, db_names = load_database_histograms()

    print("✔ Database histograms loaded:", len(db_hists))

    q_hist = compute_query_histogram(query_image, kmeans)
    if q_hist is None:
        return

    # Compute Cosine Similarity
    sims = cosine_similarity([q_hist], db_hists)[0]

    # Sort descending
    top_idx = sims.argsort()[::-1][:top_k]

    print("\n🎯 Top results:")
    for i, idx in enumerate(top_idx):
        print(f"{i+1}) {db_names[idx]}  |  similarity = {sims[idx]:.4f}")


# Run Example
if __name__ == "__main__":
    query_path = "data/queries/query1.jpg"   # ← عدليها حسب الصورة
    search(query_path)
