# Copyright 2024 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import dataclasses
from typing import ClassVar, Optional

import numpy as np


@dataclasses.dataclass
class HistogramDistanceMetric(abc.ABC):
    """
    An abstract base class for distance metrics between histograms.

    :args distrib_t: array of distribution t (usually the latest dataset distribution)
    :args distrib_u: array of distribution u (usually the sample dataset distribution)

    Each distribution must contain nonnegative floats that sum up to 1.0.
    """

    distrib_t: np.ndarray
    distrib_u: np.ndarray

    NAME: ClassVar[str]

    # noinspection PyMethodOverriding
    def __init_subclass__(cls, *, metric_name: str, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.NAME = metric_name

    @abc.abstractmethod
    def compute(self) -> float:
        raise NotImplementedError


class TotalVarianceDistance(HistogramDistanceMetric, metric_name="tvd"):
    """
    Provides a symmetric drift distance between two periods t and u
    Z - vector of random variables
    Pt - Probability distribution over time span t
    """

    def compute(self) -> float:
        """
        Calculate Total Variance distance.

        :returns:  Total Variance Distance.
        """
        return np.sum(np.abs(self.distrib_t - self.distrib_u)) / 2


class HellingerDistance(HistogramDistanceMetric, metric_name="hellinger"):
    """
    Hellinger distance is an f divergence measure, similar to the Kullback-Leibler (KL) divergence.
    It used to quantify the difference between two probability distributions.
    However, unlike KL Divergence the Hellinger divergence is symmetric and bounded over a probability space.
    The output range of Hellinger distance is [0,1]. The closer to 0, the more similar the two distributions.
    """

    def compute(self) -> float:
        """
        Calculate Hellinger Distance

        :returns: Hellinger Distance
        """
        return np.sqrt(
            max(
                1 - np.sum(np.sqrt(self.distrib_u * self.distrib_t)),
                0,  # numerical errors may produce small negative numbers, e.g. -1e-16.
                # However, Cauchy-Schwarz inequality assures this number is in the range [0, 1]
            )
        )


class KullbackLeiblerDivergence(HistogramDistanceMetric, metric_name="kld"):
    """
    KL Divergence (or relative entropy) is a measure of how one probability distribution differs from another.
    It is an asymmetric measure (thus it's not a metric) and it doesn't satisfy the triangle inequality.
    KL Divergence of 0, indicates two identical distributions.
    """

    @staticmethod
    def _calc_kl_div(
        actual_dist: np.ndarray, expected_dist: np.ndarray, zero_scaling: float
    ) -> float:
        """Return the asymmetric KL divergence"""
        # We take 0*log(0) == 0 for this calculation
        mask = actual_dist != 0
        actual_dist = actual_dist[mask]
        expected_dist = expected_dist[mask]
        with np.errstate(over="ignore"):
            # Ignore overflow warnings when dividing by small numbers,
            # resulting in inf:
            # RuntimeWarning: overflow encountered in true_divide
            relative_prob = actual_dist / np.where(
                expected_dist != 0, expected_dist, zero_scaling
            )
        return np.sum(actual_dist * np.log(relative_prob))

    def compute(
        self, capping: Optional[float] = None, zero_scaling: float = 1e-4
    ) -> float:
        """
        :param capping:      A bounded value for the KL Divergence. For infinite distance, the result is replaced with
                             the capping value which indicates a huge differences between the distributions.
        :param zero_scaling: Will be used to replace 0 values for executing the logarithmic operation.

        :returns: symmetric KL Divergence
        """
        t_u = self._calc_kl_div(self.distrib_t, self.distrib_u, zero_scaling)
        u_t = self._calc_kl_div(self.distrib_u, self.distrib_t, zero_scaling)
        result = t_u + u_t
        if capping and result == float("inf"):
            return capping
        return result
