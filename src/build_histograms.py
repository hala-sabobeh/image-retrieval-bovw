import os
import numpy as np
import joblib

# Paths
DESC_FOLDER = "features/descriptors/"
MODEL_FOLDER = "features/models/"
HIST_FOLDER = "features/histograms/"

os.makedirs(HIST_FOLDER, exist_ok=True)

# Load KMeans model
kmeans = joblib.load(os.path.join(MODEL_FOLDER, "kmeans_model.pkl"))
K = kmeans.n_clusters


def build_histogram(desc):
    """Assign descriptors to nearest cluster & build histogram."""
    words = kmeans.predict(desc)
    hist, _ = np.histogram(words, bins=np.arange(K+1), density=True)
    return hist


def main():
    files = [f for f in os.listdir(DESC_FOLDER) if f.endswith(".npy")]
    print("Total images:", len(files))

    for fname in files:
        desc_path = os.path.join(DESC_FOLDER, fname)
        desc = np.load(desc_path)

        hist = build_histogram(desc)

        save_name = fname.replace(".npy", "_hist.npy")
        save_path = os.path.join(HIST_FOLDER, save_name)

        np.save(save_path, hist)

        print("✔ Histogram saved for:", fname)

    print("\n🎉 Histograms created for all images!")


if __name__ == "__main__":
    main()
