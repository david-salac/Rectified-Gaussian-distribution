from typing import Tuple, Union

import numpy as np
from scipy.special import erf


class RectifiedGaussianDistribution(object):
    """Implementation of the rectified Gaussian distribution.

    To see what is the rectified Gaussian distribution, visit:
        https://en.wikipedia.org/wiki/Rectified_Gaussian_distribution

    Attributes:
        RTOL (float): relative tolerance for masks.
        ATOL (float): absolute tolerance for masks.
    """

    # tolerances to use when fixing numerical precision
    RTOL = 0.0
    ATOL = 0.001

    @classmethod
    def rectified_gaussian(cls,
                           mu: np.ndarray,
                           std: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Application of the rectified Gaussian distribution for lower bound
        equal to zero.

        The resulting probability distributions are rectified Gaussians
            see https://en.wikipedia.org/wiki/Rectified_Gaussian_distribution

        Args:
            mu (np.ndarray): Vector of mean values.
            std (np.ndarray): Vector of standard deviations.

        Notes:
            The resulting distributions are NOT normal.

        Returns:
            Tuple[np.ndarray, np.ndarray]: tuple of numpy arrays
                (mu, std) of resulting distributions.
        """
        if np.all(np.isnan(std)):
            # uncertainty is not known
            # fall back to deterministic clip
            _mu = np.clip(mu, 0, None)
            _std = std.copy()
            return _mu, _std

        # clip boundary
        a = np.zeros_like(mu)

        # transformed constraint
        c = (a - mu) / std

        # simplified from reference as upper limit b=inf here
        mu_t = (
            1 / np.sqrt(2 * np.pi) * np.exp(-c ** 2 / 2)
            + c / 2 * (1 + erf(c / np.sqrt(2)))
        )

        # again, simplified from reference as upper limit b=inf here
        std_t_sq = (
            (mu_t ** 2 + 1) / 2 * (1 - erf(c / np.sqrt(2)))
            + 1 / np.sqrt(2 * np.pi) * (c - 2 * mu_t) * np.exp(-c ** 2 / 2)
            + (c - mu_t) ** 2 / 2 * (1 + erf(c / np.sqrt(2)))
        )

        # we're going to sqrt std_t_sq later so we don't want marginally
        #   negative numbers fouling that operation

        # large negative numbers are erroneous and should raise an exception,
        #   so we can ignore them here

        # create two bool arrays:
        #   - is the value negative?
        #   - is it within atol of 0?
        # logically AND these two bool arrays
        small_negative_mask = (std_t_sq < 0) & np.isclose(std_t_sq,
                                                          0.0,
                                                          rtol=cls.RTOL,
                                                          atol=cls.ATOL)

        # use the resulting mask to set these values to 0
        std_t_sq[small_negative_mask] = 0.0

        _mu = mu + std * mu_t
        _std = std * np.sqrt(std_t_sq)

        # fix slight violations of limits (due to numerical precision)
        mask = (_mu < 0) & np.isclose(_mu, 0, rtol=cls.RTOL, atol=cls.ATOL)
        _mu[mask] = 0.0

        return _mu, _std

    @classmethod
    def rectified_gaussian_general_bound(cls,
                                         mu: np.ndarray, std: np.ndarray,
                                         lower: Union[np.ndarray, float],
                                         upper: Union[np.ndarray, float]
                                         ) -> Tuple[np.ndarray, np.ndarray]:
        """Application of the rectified Gaussian distribution for gneral lower
            and upper bounds (including infinity).

        The resulting probability distributions are rectified Gaussians
            see https://en.wikipedia.org/wiki/Rectified_Gaussian_distribution

        Args:
            lower (Union[np.ndarray, float]): Lower bound or bounds, also
                accepts float('-inf') option to reflect only the upper bound.
            upper (Union[np.ndarray, float]): Upper bound or bounds, also
                accepts float('+inf') option to reflect only the lower bound.
            mu (np.ndarray): Vector of mean values.
            std (np.ndarray): Vector of standard deviations.

        Notes:
            The resulting distributions are NOT normal.

        Returns:
            Tuple[np.ndarray, np.ndarray]: tuple of numpy arrays
                (mu, std) of resulting distributions.
        """

        if np.all(np.isnan(std)):
            # uncertainty is not known
            # fall back to deterministic clip
            _mu = np.clip(mu, lower, upper)
            _std = std.copy()
            return _mu, _std

        # clip boundary
        a = lower
        b = upper

        # transformed constraint
        c = (a - mu) / std
        d = (b - mu) / std

        # general version from reference
        mu_t = (
            1 / np.sqrt(2 * np.pi) * (
                np.exp(-c ** 2 / 2)
                - np.exp(-d ** 2 / 2)
            )
            + c / 2 * (1 + erf(c / np.sqrt(2)))
            + d / 2 * (1 - erf(d / np.sqrt(2)))
        )
        if isinstance(lower, float) and lower == float('-inf'):
            # because of: lim n-> -inf (n * exp(-n^2)) = 0
            mu_t = (
                1 / np.sqrt(2 * np.pi) * (
                    - np.exp(-d ** 2 / 2)
                )
                + d / 2 * (1 - erf(d / np.sqrt(2)))
            )
        if isinstance(upper, float) and upper == float('+inf'):
            # because of: lim n-> inf (-n * exp(-n^2)) = 0
            mu_t = (
                1 / np.sqrt(2 * np.pi) * (
                    np.exp(-c ** 2 / 2)
                )
                + c / 2 * (1 + erf(c / np.sqrt(2)))
            )

        # general version from reference
        std_t_sq = (
            (mu_t ** 2 + 1) / 2 * (
                erf(d / np.sqrt(2))
                - erf(c / np.sqrt(2))
            )
            - 1 / np.sqrt(2 * np.pi) * (
                (d - 2 * mu_t) * np.exp(-d ** 2 / 2)
                - (c - 2 * mu_t) * np.exp(-c ** 2 / 2)
            )
            + (c - mu_t) ** 2 / 2 * (1 + erf(c / np.sqrt(2)))
            + (d - mu_t) ** 2 / 2 * (1 - erf(d / np.sqrt(2)))
        )
        if isinstance(lower, float) and lower == float('-inf'):
            # (for Q, K, J const in R):
            #   lim c->-inf (-Q * erf(c) + (K * c * exp(-c^2))
            #       + J * c^2 * (1 + erf(c))) = Q
            std_t_sq = (
                (mu_t ** 2 + 1) / 2 * (
                    erf(d / np.sqrt(2))
                    + 1
                )
                - 1 / np.sqrt(2 * np.pi) * (
                    (d - 2 * mu_t) * np.exp(-d ** 2 / 2)
                )
                + (d - mu_t) ** 2 / 2 * (1 - erf(d / np.sqrt(2)))
            )

        if isinstance(upper, float) and upper == float('+inf'):
            # (for Q, K, J const in R):
            #   lim d->inf (Q * erf(d) + (K * d * exp(-d^2)) +
            #       J * d^2 * (1 - erf(d))) = Q
            std_t_sq = (
                (mu_t ** 2 + 1) / 2 * (
                    1
                    - erf(c / np.sqrt(2))
                )
                - 1 / np.sqrt(2 * np.pi) * (
                    - (c - 2 * mu_t) * np.exp(-c ** 2 / 2)
                )
                + (c - mu_t) ** 2 / 2 * (1 + erf(c / np.sqrt(2)))
            )

        # To check the values of limits, try Wolfram Alpha:
        #   https://www.wolframalpha.com

        # we're going to sqrt std_t_sq later so we don't want marginally
        #   negative numbers fouling that operation

        # large negative numbers are erroneous and should raise an exception,
        #   so we can ignore them here

        # create two bool arrays:
        #   - is the value negative?
        #   - is it within atol of 0?
        # logically AND these two bool arrays
        small_negative_mask = (std_t_sq < 0) & np.isclose(std_t_sq,
                                                          0.0,
                                                          rtol=cls.RTOL,
                                                          atol=cls.ATOL)

        # use the resulting mask to set these values to 0
        std_t_sq[small_negative_mask] = 0

        _mu = mu + std * mu_t
        _std = std * np.sqrt(std_t_sq)

        # fix slight violations of limits (due to numerical precision)

        mask = (_mu < lower) & np.isclose(_mu,
                                          lower,
                                          rtol=cls.RTOL,
                                          atol=cls.ATOL)
        if isinstance(lower, np.ndarray):
            _mu[mask] = lower[mask]
        else:
            _mu[mask] = lower

        mask = (_mu > upper) & np.isclose(_mu,
                                          upper,
                                          rtol=cls.RTOL,
                                          atol=cls.ATOL)
        if isinstance(upper, np.ndarray):
            _mu[mask] = upper[mask]
        else:
            _mu[mask] = upper

        return _mu, _std
