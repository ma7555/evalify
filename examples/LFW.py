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
print('Starting Experiment')
experiment.run(
    X,
    y,
    metrics=(
        "cosine_similarity",
        "pearson_similarity",
        "euclidean_distance_l2",
    ),
    same_class_samples="full",
    different_class_samples=("full", "full"),
)
print(
    f"Total available embeddings {len(y)} resulted in {len(experiment.df)} "
    "samples for the experiment."
)
print(f"Metrics calculations executed in {time.time()-start_time:.2f} seconds")
print("ROC AUC:")
print(experiment.get_roc_auc())
print("Threshold @ FPR:")
print(experiment.find_threshold_at_fpr(0.01))
print(experiment.calculate_eer())
