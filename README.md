# evalify

<p align="center">

<img src="https://user-images.githubusercontent.com/7144929/154332210-fa1fee34-faae-4567-858a-49fa53e99a2b.svg" width="292" height="120" alt="Logo"/>

</p>

<p align="center">

<a href="https://github.com/ma7555/evalify/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/ma7555/evalify"
        alt = "License">
</a>
<a href="https://doi.org/10.5281/zenodo.6181723"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.6181723.svg" alt="DOI"></a>
<a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.7 | 3.8 | 3.9 | 3.10-blue.svg"
        alt = "Python 3.7 | 3.8 | 3.9 | 3">
</a>
<a href="https://pypi.python.org/pypi/evalify">
    <img src="https://img.shields.io/pypi/v/evalify.svg"
        alt = "Release Status">
</a>
<a href="https://github.com/ma7555/evalify/actions">
    <img src="https://github.com/ma7555/evalify/actions/workflows/dev.yml/badge.svg?branch=main" alt="CI Status">
</a>
<a href="https://ma7555.github.io/evalify/">
    <img src="https://img.shields.io/website/https/ma7555.github.io/evalify/index.html.svg?label=docs&down_message=unavailable&up_message=available" alt="Documentation Status">
</a>
<a href="https://github.com/psf/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black">
</a>
<a href="https://codecov.io/gh/ma7555/evalify">
  <img src="https://codecov.io/gh/ma7555/evalify/branch/main/graph/badge.svg" />
</a>
<a href="https://github.com/ma7555/evalify/releases"><img alt="GitHub all releases" src="https://img.shields.io/github/downloads/ma7555/evalify/total">
</a>

</p>


Evaluate your face or voice verification models literally in seconds.

## Installation
#### Stable release
```bash
pip install evalify
```
#### Bleeding edge
* From source
    ```bash
    pip install git+https://github.com/ma7555/evalify.git
    ```
* From TestPyPI
    ```bash
    pip install --index-url https://test.pypi.org/simple/ evalify
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

* Blazing fast implementation for metrics calculation through optimized einstein sum and vectorized calculations.
* Many operations are dispatched to canonical BLAS, cuBLAS, or other specialized routines.
* Smart sampling options using direct indexing from pre-calculated arrays with an option to have total control over sampling strategy and sampling numbers.
* Supports most evaluation metrics:
    - cosine_similarity
    - cosine_distance
    - euclidean_distance
    - euclidean_distance_l2
    - minkowski_distance
    - manhattan_distance
    - chebyshev_distance
* Computation time for 4 metrics 4.2 million samples experiment is **24 seconds vs 51 minutes** if looping using `scipy.spatial.distance` implemntations.

## TODO
* Safer memory allocation. I did not have issues but if you ran out of memory please manually increase number of splits with `nsplits` argument.

## Contribution
* Contributions are welcomed, and they are greatly appreciated! Every little bit helps, and credit will always be given.
* Please check [CONTRIBUTING.md](https://github.com/ma7555/evalify/blob/main/CONTRIBUTING.md) for guidelines.

## Citation
* If you use this software, please cite it using the metadata from [CITATION.cff](https://github.com/ma7555/evalify/blob/main/CITATION.cff)

