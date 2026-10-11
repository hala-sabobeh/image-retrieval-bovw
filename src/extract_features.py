import os
import cv2
import numpy as np

# Path to dataset
IMAGES_FOLDER = "data/images/visionset/Paris6k/"

# Folder to save SIFT descriptors
DESC_SAVE_FOLDER = "features/descriptors/"

# Create output folder if not exists
os.makedirs(DESC_SAVE_FOLDER, exist_ok=True)


def load_images(folder):
    """Recursively load all images inside all subfolders."""
    image_paths = []
    for root, dirs, files in os.walk(folder):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(root, fname)
                image_paths.append(full_path)
    return image_paths


def extract_and_save_sift(img_path, sift):
    """Extract SIFT descriptors for one image and save them."""

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("⚠ Could not read:", img_path)
        return

    keypoints, descriptors = sift.detectAndCompute(img, None)

    if descriptors is None:
        print("⚠ No SIFT descriptors for:", img_path)
        return

    # Save as .npy file
    filename = os.path.basename(img_path)
    file_no_ext = os.path.splitext(filename)[0]
    save_path = os.path.join(DESC_SAVE_FOLDER, file_no_ext + ".npy")

    np.save(save_path, descriptors)
    print("✔ Saved SIFT for:", filename)


def main():
    images = load_images(IMAGES_FOLDER)
    print("🔹 Total images to process:", len(images))

    # Create SIFT object
    sift = cv2.SIFT_create()

    # Loop through all images
    for img_path in images:
        extract_and_save_sift(img_path, sift)

    print("\n🎉 SIFT feature extraction completed!")


if __name__ == "__main__":
    main()
