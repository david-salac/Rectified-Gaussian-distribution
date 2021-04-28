# Implementation of the rectified Gaussian distribution
This is a Python implementation of Rectified Gaussian distribution using SciPy.
Contains standard rectified Gaussian distribution as well as enhanced version
with general bounds (for upper and lower clip).

Author: David Salac <https://github.com/david-salac>

## Mathematics behind
Read more about the rectified Gaussian distribution on Wikipedia:

* https://en.wikipedia.org/wiki/Rectified_Gaussian_distribution

this page is also a source for the implementation.

### About the rectified Gaussian distribution
In probability theory, the rectified Gaussian distribution is
a modification of the Gaussian distribution when its negative
elements are reset to 0 (analogous to an electronic rectifier).
It is essentially a mixture of a discrete distribution (constant 0)
and a continuous distribution (a truncated Gaussian distribution with
the interval (0 ,∞) as a result of censoring.

### Rectified Gaussian distribution with general bounds
The special application of rectified Gaussian distribution where
both lower and upper bounds are flexible (not just 0 for lower
bound and infinity ∞ for the upper one).

## Usage
To use the script, just copy file `rectified_gaussian.py` to your source code.

To use standard rectified Gaussian distribution, follow:
```python
from .rectified_gaussian import RectifiedGaussianDistribution
# ...
mu: np.ndarray = ...  # Some value for mean
std: np.ndarray = ...  # Some value for standard deviation
rectified_mu, rectified_std = \
    RectifiedGaussianDistribution.rectified_gaussian(mu, std)
```

To use standard rectified Gaussian distribution with general bounds, follow:
```python
from .rectified_gaussian import RectifiedGaussianDistribution
# ...
mu: np.ndarray = ...  # Some value for mean
std: np.ndarray = ...  # Some value for standard deviation
lower_bound: Union[np.ndarray, float] = ... # Some value(s) for lower bound
upper_bound: Union[np.ndarray, float] = ... # Some value(s) for upper bound

rectified_mu, rectified_std = \
    RectifiedGaussianDistribution.rectified_gaussian_general_bound(
        mu, std, lower_bound, upper_bound
    )

# Note, lower bound can be -inf and upper bound +inf to avoid using them:
lower_bound = float('-inf')  # To skip lower bound
upper_bound = float('+inf')  # To skip upper bound
```
