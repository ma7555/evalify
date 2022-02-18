# evalify

<p align="center">

<img src="https://user-images.githubusercontent.com/7144929/154332210-fa1fee34-faae-4567-858a-49fa53e99a2b.svg" width="292" height="120" alt="Logo"/>

</p>

<p align="center">

<a href="https://pypi.python.org/pypi/evalify">
    <img src="https://img.shields.io/pypi/v/evalify.svg"
        alt = "Release Status">
</a>

<a href="https://github.com/ma7555/evalify/actions">
    <img src="https://github.com/ma7555/evalify/actions/workflows/dev.yml/badge.svg?branch=main" alt="CI Status">
</a>

<a href="https://evalify.readthedocs.io/en/latest/?badge=latest">
    <img src="https://readthedocs.org/projects/evalify/badge/?version=latest" alt="Documentation Status">
</a>

<a href="https://github.com/psf/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black">
</a>

</p>


evalify contains tools needed to evaluate your face verification models literally in seconds.

## Installation
#### Stable release
```bash
pip install evalify
```
#### Bleeding edge
```bash
pip install git+https://github.com/ma7555/evalify.git
```

## Usage

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

## Documentation: 
* <https://ma7555.github.io/evalify/>


## Features

* Blazing fast implementation for metrics calculation through optimized einstein sum.
* Many operations are dispatched to canonical BLAS, cuBLAS, or other specialized routines.
* Smart sampling options using direct indexing from pre-calculated arrays.
* Supports common evaluation metrics like cosine similarity, euclidean distance and l2 normalized euclidean distance.

## Contribution
* Contributions are welcomed, and they are greatly appreciated! Every little bit helps, and credit will always be given.
* Please check [CONTRIBUTING.md](https://github.com/ma7555/evalify/blob/main/CONTRIBUTING.md) for guidelines.

## License
* [BSD-3-Clause](https://github.com/ma7555/evalify/blob/main/LICENSE)

## Citation
* If you use this software, please cite it using the metadata from [CITATION.cff](https://github.com/ma7555/evalify/blob/main/CITATION.cff)

