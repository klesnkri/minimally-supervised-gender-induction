from collections import Counter, defaultdict


def normalize_probability_distribution(prob_distribution: dict | defaultdict | Counter) -> dict | defaultdict | Counter:
    """
    Normalize probability distribution to sum to 1.
    :param prob_distribution: Probability distribution to be normalized
    :return: Normalized probability distribution
    """
    sum_probs = sum(prob_distribution.values())

    # All probabilities 0
    if sum_probs == 0:
        return prob_distribution

    # Normalize
    for key in prob_distribution:
        prob_distribution[key] /= sum_probs

    return prob_distribution
