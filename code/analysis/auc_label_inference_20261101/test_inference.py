import unittest
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from infer_labels import (rank_features, targets, rank_auc, conditioned_probabilities,
                          exchange_equal_score_labels, weighted_negative_candidates)


class InferenceTests(unittest.TestCase):
    def test_rank_identity_with_tied_scores(self):
        rng = np.random.default_rng(41)
        prediction = np.round(rng.random((300, 4)), 1)
        labels = np.zeros(300, dtype="int8")
        labels[rng.choice(300, 37, replace=False)] = 1
        observed = rank_auc(labels, rank_features(prediction))
        np.testing.assert_allclose(observed, [roc_auc_score(labels, prediction[:, j]) for j in range(4)], atol=1e-14, rtol=0)

    def test_projection_recovers_known_interior_moments(self):
        rng = np.random.default_rng(18)
        predictions = rng.random((1500, 4))
        u = rank_features(predictions)
        prior = .005 + .025 * rng.random(1500)
        positives = 40
        offset = logit(prior) + u @ np.array([.8, -.4, .3, .5])
        shift = brentq(lambda value: expit(offset + value).sum() - positives, -10, 10)
        truth_probability = expit(offset + shift)
        aggregate = .5 + len(prior) * (truth_probability @ u) / (positives * (len(prior) - positives))
        probability, _, info = conditioned_probabilities(prior, u, aggregate, positives)
        self.assertLess(abs(probability.sum() - positives), .001)
        np.testing.assert_allclose(probability, truth_probability, atol=1e-7, rtol=1e-6)
        self.assertLess(info["maximum_expected_rank_moment_error"], 1e-7)

    def test_equal_score_swaps_change_labels_without_any_auc_change(self):
        values = np.array([[.1, .2, .3, .4], [.1, .2, .3, .4], [.7, .8, .9, .9], [.7, .8, .9, .9], [.4, .5, .6, .7]])
        labels = np.array([1, 0, 0, 1, 0], dtype="int8")
        _, groups = np.unique(values, axis=0, return_inverse=True)
        other = exchange_equal_score_labels(labels, groups)
        self.assertGreater(int((other != labels).sum()), 0)
        self.assertEqual(int(other.sum()), int(labels.sum()))
        np.testing.assert_array_equal(rank_auc(labels, rank_features(values)), rank_auc(other, rank_features(values)))

    def test_proposal_candidates_exclude_current_positives(self):
        probability = np.linspace(.001, .1, 200)
        selected = np.zeros(200, dtype=bool)
        selected[:17] = True
        ids = weighted_negative_candidates(np.random.default_rng(4), probability, selected, 512)
        self.assertFalse(selected[ids].any())
        self.assertEqual(len(ids), len(np.unique(ids)))


if __name__ == "__main__":
    unittest.main()
