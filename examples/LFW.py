""" File LFW.npz contains sample embeddings and targets from LFW dataset"""

from pathlib import Path
import time
import numpy as np

from evalify import Experiment

lfw_npz = Path(__file__).parent.parent / Path("tests/data/LFW.npz")
X_y_array = np.load(lfw_npz)
X = X_y_array["X"][:1000]
y = X_y_array["y"][:1000]

experiment = Experiment(
    metrics=(
        "cosine_similarity",
        "pearson_similarity",
        "euclidean_distance_l2",
    ),
    same_class_samples="full",
    different_class_samples=("full", "full"),
)
start_time = time.time()
print("Starting Experiment")
experiment.run(X, y)
print(
    f"Total available embeddings {len(y)} resulted in {len(experiment.df)} "
    "samples for the experiment."
)
print(f"Metrics calculations executed in {time.time()-start_time:.2f} seconds")
print("ROC AUC:")
print(experiment.roc_auc())
print("threshold @ FPR:")
print(experiment.threshold_at_fpr(0.01))
print("EER:")
print(experiment.eer())
print("TAR@FAR:")
print(experiment.tar_at_far([0.01, 0.001]))
