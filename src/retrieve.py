import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
import time
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- Paths ----------------
HIST_FOLDER = "../features/histograms/"
MODEL_FOLDER = "../features/models/"
QUERY_FOLDER = "../data/queries/"
DATASET_FOLDER = "../data/images/visionset/Paris6k/"

# ---------------- Helper ----------------
def get_label(name):
    return name.split("_")[1]

# ---------------- Load models ----------------
def load_models():
    print("Loading KMeans model...")
    kmeans = joblib.load(os.path.join(MODEL_FOLDER, "kmeans_model.pkl"))

    print("Loading IDF...")
    idf = np.load(os.path.join(MODEL_FOLDER, "idf.npy"))

    return kmeans, idf

# ---------------- TF-IDF ----------------
def apply_tfidf(hist, idf):
    tfidf = hist * idf
    tfidf = tfidf / (np.linalg.norm(tfidf) + 1e-8)
    return tfidf

# ---------------- Build query histogram ----------------
def build_query_histogram(image_path, kmeans, idf):
    sift = cv2.SIFT_create()
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Cannot read query image!")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, des = sift.detectAndCompute(gray, None)

    K = kmeans.n_clusters
    if des is None:
        hist = np.zeros(K)
    else:
        labels = kmeans.predict(des)
        hist, _ = np.histogram(labels, bins=np.arange(K + 1), density=True)

    return apply_tfidf(hist, idf)

# ---------------- Load database histograms ----------------
def load_database_histograms(idf):
    print("Loading database histograms...")
    hists, names = [], []

    for fname in os.listdir(HIST_FOLDER):
        if fname.endswith(".npy"):
            hist = np.load(os.path.join(HIST_FOLDER, fname))
            hist = apply_tfidf(hist, idf)
            hists.append(hist)
            names.append(fname.replace("_hist.npy", ".jpg"))

    print("Loaded:", len(hists), "histograms")
    return np.array(hists), names

# ---------------- Evaluation Metrics ----------------
def precision_at_k(retrieved, query_label, k):
    correct = sum(1 for i in range(k) if get_label(retrieved[i]) == query_label)
    return correct / k

def recall_at_k(retrieved, query_label, total_relevant, k):
    correct = sum(1 for i in range(k) if get_label(retrieved[i]) == query_label)
    return correct / total_relevant if total_relevant > 0 else 0

def average_precision(retrieved, query_label):
    score, hits = 0, 0
    for i, name in enumerate(retrieved):
        if get_label(name) == query_label:
            hits += 1
            score += hits / (i + 1)
    return score / hits if hits else 0

# ---------------- Visualization ----------------
def find_image_path(image_name):
    for root, _, files in os.walk(DATASET_FOLDER):
        if image_name in files:
            return os.path.join(root, image_name)
    return None

def visualize(query_path, retrieved_names, query_label, cols=5):
    total = len(retrieved_names) + 1
    rows = int(np.ceil(total / cols))

    plt.figure(figsize=(4 * cols, 4 * rows))

    qimg = cv2.cvtColor(cv2.imread(query_path), cv2.COLOR_BGR2RGB)
    plt.subplot(rows, cols, 1)
    plt.imshow(qimg)
    plt.title("Query Image")
    plt.axis("off")

    for i, name in enumerate(retrieved_names):
        img_path = find_image_path(name)
        if img_path is None:
            continue

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        plt.subplot(rows, cols, i + 2)
        plt.imshow(img)

        label = get_label(name)
        color = "green" if label == query_label else "red"
        plt.title(f"Rank {i+1}\n({label})", color=color)
        plt.axis("off")

    plt.tight_layout(h_pad=3.0, w_pad=2.0)
    plt.show()

# ---------------- Main ----------------
def main():
    query_name = input("Enter query image name: ")
    top_k = int(input("Enter number of top matches: "))

    query_label = get_label(query_name)
    query_path = os.path.join(QUERY_FOLDER, query_name)

    kmeans, idf = load_models()
    db_hists, names = load_database_histograms(idf)

    # -------- Start timing --------
    start_time = time.time()

    query_hist = build_query_histogram(query_path, kmeans, idf)
    if query_hist is None:
        return

    sims = cosine_similarity(query_hist.reshape(1, -1), db_hists)[0]
    idx = sims.argsort()[::-1][:top_k]
    retrieved = [names[i] for i in idx]

    retrieval_time = time.time() - start_time
    # -------- End timing --------

    print("\nTop Results:")
    for i, name in enumerate(retrieved):
        print(f"{i+1}) {name} | similarity = {sims[idx[i]]:.4f}")

    # Count total relevant images in database
    total_relevant = sum(1 for n in names if get_label(n) == query_label)

    p_at_k = precision_at_k(retrieved, query_label, top_k)
    r_at_k = recall_at_k(retrieved, query_label, total_relevant, top_k)
    ap = average_precision(retrieved, query_label)

    print("\nEvaluation:")
    print(f"Precision@{top_k}: {p_at_k:.3f}")
    print(f"Recall@{top_k}: {r_at_k:.3f}")
    print(f"Average Precision (AP): {ap:.3f}")
    print(f"Retrieval Time: {retrieval_time:.3f} seconds")

    visualize(query_path, retrieved, query_label)

if __name__ == "__main__":
    main()
