""" File LFW.npz contains sample embeddings and targets from LFW dataset"""

import os
import pathlib
import time

import numpy as np

from evalify import Experiment

lfw_npz = os.path.join(pathlib.Path(__file__).parent, "LFW.npz")
X_y_array = np.load(lfw_npz)
X = X_y_array["X"]
y = X_y_array["y"]

experiment = Experiment()
start_time = time.time()
experiment.run(
    X,
    y,
    metrics=(
        "cosine_distance",
        "euclidean_distance",
        "manhattan_distance",
        "euclidean_distance_l2",
    ),
    same_class_samples="full",
    different_class_samples=("full", "full"),
    nsplits=400,
)

print(
    f"Metrics calculations executed in {time.time()-start_time:.2f} seconds for processes"
)

print(
    f"Total available embeddings {len(y)} resulted in {len(experiment.df)} "
    "samples for the experiment."
)
# print("ROC AUC:")
# print(experiment.get_roc_auc())
