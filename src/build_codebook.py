import os
import numpy as np
from sklearn.cluster import KMeans
import joblib

# Path where SIFT descriptors are stored
DESC_FOLDER = "features/descriptors/"

# Where to save codebook and model
MODEL_FOLDER = "features/models/"
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Number of clusters = size of codebook
K = 500

# Maximum number of descriptors to sample
MAX_DESCRIPTORS = 100000


def load_all_descriptors(desc_folder):
    """Load and sample descriptors from all .npy files."""
    all_desc = []

    for fname in os.listdir(desc_folder):
        if fname.endswith(".npy"):
            desc_path = os.path.join(desc_folder, fname)
            desc = np.load(desc_path)

            # Add descriptors to the list
            all_desc.append(desc)

    # Concatenate all descriptors
    all_desc = np.vstack(all_desc)

    print("Total descriptors before sampling:", all_desc.shape)

    # If too many descriptors → sample randomly
    if all_desc.shape[0] > MAX_DESCRIPTORS:
        idx = np.random.choice(all_desc.shape[0], MAX_DESCRIPTORS, replace=False)
        all_desc = all_desc[idx]

    print("Descriptors used for K-means:", all_desc.shape)
    return all_desc


def main():
    print("🔹 Loading descriptors...")
    descriptors = load_all_descriptors(DESC_FOLDER)

    print("🔹 Running K-means with K =", K)
    kmeans = KMeans(n_clusters=K, random_state=42)
    kmeans.fit(descriptors)

    # Save model
    joblib.dump(kmeans, os.path.join(MODEL_FOLDER, "kmeans_model.pkl"))

    # Save codebook (cluster centers)
    codebook = kmeans.cluster_centers_
    np.save(os.path.join(MODEL_FOLDER, "codebook.npy"), codebook)

    print("\n🎉 Codebook created successfully!")
    print("✔ Saved to:", MODEL_FOLDER)


if __name__ == "__main__":
    main()
