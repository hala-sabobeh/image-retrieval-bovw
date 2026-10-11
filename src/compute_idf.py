import os
import numpy as np

HIST_FOLDER = "../features/histograms/"
SAVE_PATH = "../features/models/idf.npy"

hists = []

for f in os.listdir(HIST_FOLDER):
    if f.endswith(".npy"):
        hists.append(np.load(os.path.join(HIST_FOLDER, f)))

hists = np.array(hists)   # (N_images, K)

N = hists.shape[0]
df = np.count_nonzero(hists > 0, axis=0)

idf = np.log((N + 1) / (df + 1))
np.save(SAVE_PATH, idf)

print("IDF saved:", idf.shape)



