import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

DATASET_FOLDER = "../data/images/visionset/Paris6k/"
IMAGES_PER_ROW = 5

# ---------------- Helpers ----------------
def get_label(name):
    return name.split("_")[1]

def find_image_path(image_name):
    for root, _, files in os.walk(DATASET_FOLDER):
        if image_name in files:
            return os.path.join(root, image_name)
    return None

# ---------------- Visualization ----------------
def visualize_results(query_path, retrieved_names, query_label):
    total_images = len(retrieved_names) + 1
    rows = int(np.ceil(total_images / IMAGES_PER_ROW))

    plt.figure(figsize=(20, 5 * rows))

    # -------- Query image --------
    query_img = cv2.cvtColor(cv2.imread(query_path), cv2.COLOR_BGR2RGB)
    ax = plt.subplot(rows, IMAGES_PER_ROW, 1)
    ax.imshow(query_img)
    ax.set_title("Query Image", fontsize=14, pad=15)
    ax.axis("off")

    # -------- Retrieved images --------
    for i, name in enumerate(retrieved_names):
        img_path = find_image_path(name)
        if img_path is None:
            continue

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        ax = plt.subplot(rows, IMAGES_PER_ROW, i + 2)
        ax.imshow(img)

        label = get_label(name)
        is_correct = (label == query_label)

        title_color = "darkgreen" if is_correct else "darkred"

        ax.set_title(
            f"Rank {i+1}\n({label})",
            color=title_color,
            fontsize=11,
            pad=18      
        )
        ax.axis("off")

    # -------- Global spacing --------
    plt.subplots_adjust(
        hspace=0.6,   
        wspace=0.35   
    )

    plt.show()
