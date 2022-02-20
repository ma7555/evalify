""" File LFW.npz contains sample embeddings and targets from LFW dataset"""

import numpy as np
from evalify.evalify import Experiment
import time


X_y_array = np.load("./examples/LFW.npz")
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
)
print(
    f"Total available embeddings {len(y)} resulted in {len(experiment.df)} "
    "samples for the experiment."
)
print(f"Metrics calculations executed in {time.time()-start_time:.2f} seconds")
print("ROC AUC:")
print(experiment.get_roc_auc())
