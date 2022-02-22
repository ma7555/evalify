# Usage

To use evalify in a project

```python
import numpy as np
from evalify import Experiment

rng = np.random.default_rng()
nphotos = 500
emb_size = 32
nclasses = 10
X = rng.random((self.nphotos, self.emb_size))
y = rng.integers(self.nclasses, size=self.nphotos)

experiment = Experiment()
experiment.run(X, y)
experiment.get_roc_auc()
print(experiment.df.roc_auc)
```

For a working experiment using real face embeddings, please refer to `LFW.py` under `./examples`.

```python
python ./examples/LFW.py
```
```
Total available embeddings 2921 resulted in 4264660 samples for the experiment.
Metrics calculations executed in 24.05 seconds
ROC AUC:
OrderedDict([('euclidean_distance', 0.9991302819624498), ('cosine_distance', 0.9991302818953706), ('euclidean_distance_l2', 0.9991302818953706), ('manhattan_distance', 0.9991260462584446)])
```
