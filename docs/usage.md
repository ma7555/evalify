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

