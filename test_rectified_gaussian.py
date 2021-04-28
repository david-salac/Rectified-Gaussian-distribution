from unittest import TestCase

from scipy.stats import norm
import numpy as np

from .rectified_gaussian import RectifiedGaussianDistribution


def sample_clip(mu_vec, std_vec, lower=0, upper=np.inf, nsamp=1_000_000):
    """empirical version of clip"""
    mu = np.zeros(len(mu_vec))
    std = np.zeros(len(mu_vec))
    for idx in range(len(mu_vec)):
        rv = norm(mu_vec[idx], std_vec[idx])
        X = rv.rvs(size=nsamp)
        # Apply threshold .clip(lower=0)
        X[X < lower] = lower
        if np.isfinite(upper):
            X[X > upper] = upper
        std[idx] = np.std(X)
        mu[idx] = np.mean(X)
    return mu, std


class Test_RectifiedGaussianDistribution(TestCase):

    def test_ok(self):
        """
        ensure analytic clip function (fast)
        can be reproduced empirically (slow)
        """

        atol = 1.e-2

        mu = np.arange(-4, 5)
        std = np.arange(1, 10)

        std = std / 2

        sample_mu, sample_std = sample_clip(mu, std)

        analytic_mu, analytic_std = \
            RectifiedGaussianDistribution.rectified_gaussian(mu, std)

        self.assertTrue(
            np.allclose(
                analytic_mu,
                sample_mu,
                atol=atol
            ),
            "mu is different {} {}".format(analytic_mu, sample_mu)
        )

        self.assertTrue(
            np.allclose(
                sample_std,
                analytic_std,
                atol=atol
            ),
            "std is different {} {}".format(analytic_std, sample_std)
        )

    def test_limit_violation(self):
        # due to numerical instability
        # these inputs previously produced a mu
        # that was outside of limits

        [mu], _ = RectifiedGaussianDistribution.rectified_gaussian(
            np.array([-1.9015964027872552]),
            np.array([0.22821715451447067])
        )

        self.assertTrue(0 <= mu)

        [mu], _ = RectifiedGaussianDistribution.rectified_gaussian(
            np.array([-1.835146232187307]),
            np.array([0.08823442696782147])
        )

        self.assertTrue(0 <= mu)

    def test_std_nan(self):

        atol = 1.e-2

        mu = np.arange(-4, 5)
        std = np.full_like(mu, np.nan, float)

        analytic_mu, analytic_std = \
            RectifiedGaussianDistribution.rectified_gaussian(mu, std)

        self.assertTrue(
            np.allclose(
                analytic_mu,
                np.clip(mu, 0, None),
                atol=atol
            )
        )

        self.assertTrue(np.all(np.isnan(analytic_std)))


class Test_clip_general_bound(TestCase):

    def test_ok(self):
        """
        ensure analytic clip function (fast)
        can be reproduced empirically (slow)
        """

        atol = 1.e-2

        mu = np.arange(-4, 5)
        std = np.arange(1, 10)

        std = std / 2

        sample_mu, sample_std = sample_clip(mu, std, -2, 2)

        analytic_mu, analytic_std = \
            RectifiedGaussianDistribution.rectified_gaussian_general_bound(
                mu, std, -2, 2
            )

        self.assertTrue(
            np.allclose(
                analytic_mu,
                sample_mu,
                atol=atol
            ),
            "mu is different {} {}".format(analytic_mu, sample_mu)
        )

        self.assertTrue(
            np.allclose(
                sample_std,
                analytic_std,
                atol=atol
            ),
            "std is different {} {}".format(analytic_std, sample_std)
        )

    def test_limit_violation(self):
        # due to numerical instability
        # these inputs previously produced a mu
        # that was outside of limits

        [mu], _ = \
            RectifiedGaussianDistribution.rectified_gaussian_general_bound(
            np.array([3.8967648016673087]),
            np.array([0.2365544570480549]),
            -2,
            2
        )

        self.assertTrue(-2 <= mu <= 2)

        [mu], _ = \
            RectifiedGaussianDistribution.rectified_gaussian_general_bound(
            np.array([3.871112940029491]),
            np.array([0.22741225380745078]),
            -2,
            2
        )

        self.assertTrue(-2 <= mu <= 2)

        [mu], _ = \
            RectifiedGaussianDistribution.rectified_gaussian_general_bound(
            np.array([-3.8003152512350917]),
            np.array([0.21914280873442937]),
            -2,
            2
        )

        self.assertTrue(-2 <= mu <= 2)

        [mu], _ = \
            RectifiedGaussianDistribution.rectified_gaussian_general_bound(
            np.array([-3.250683279821234]),
            np.array([0.15178901355449304]),
            -2,
            2
        )

        self.assertTrue(-2 <= mu <= 2)

    def test_std_nan(self):

        atol = 1.e-2

        mu = np.arange(-4, 5)
        std = np.full_like(mu, np.nan, float)

        analytic_mu, analytic_std = \
            RectifiedGaussianDistribution.rectified_gaussian_general_bound(
                mu, std, -2, 2
            )

        self.assertTrue(
            np.allclose(
                analytic_mu,
                np.clip(mu, -2, 2),
                atol=atol
            )
        )

        self.assertTrue(np.all(np.isnan(analytic_std)))
