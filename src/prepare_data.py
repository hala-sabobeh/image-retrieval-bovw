import os
import cv2

# Path to your dataset folder (Paris6k)
IMAGES_FOLDER = "data/images/visionset/Paris6k/"

# -------------------------------------------------------------------
# Recursive function: walks through all subfolders and collects images
# -------------------------------------------------------------------
def load_images(folder):
    image_paths = []

    # os.walk allows us to go through all subdirectories automatically
    for root, dirs, files in os.walk(folder):
        for fname in files:
            # Only accept image files
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(root, fname)
                image_paths.append(full_path)

    return image_paths


def main():
    # Load all images recursively
    images = load_images(IMAGES_FOLDER)

    print("🔹 Total number of images found:", len(images))

    if len(images) == 0:
        print("⚠ No images found! Check the folder path.")
        return

    # Try to read the first image
    img = cv2.imread(images[0])
    if img is None:
        print("⚠ Could not read the first image! Check file format or permissions.")
    else:
        print("✔ First image loaded successfully!")
        print("✔ Image shape (H, W, C):", img.shape)


if __name__ == "__main__":
    main()
