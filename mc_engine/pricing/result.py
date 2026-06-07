from dataclasses import dataclass


@dataclass
class PricingResult:
    """Output of a Monte Carlo pricing run.

    All values are expressed in the same units as the product's payoff
    (typically currency per unit of notional).

    Attributes
    ----------
    price:
        Monte Carlo estimate of the fair value (mean discounted payoff).
    std_error:
        Standard error of the mean: std(payoffs) / sqrt(n_sims).
    ci_lower:
        Lower bound of the 95 % confidence interval (price - 1.96 · se).
    ci_upper:
        Upper bound of the 95 % confidence interval (price + 1.96 · se).
    """

    price:     float
    std_error: float
    ci_lower:  float
    ci_upper:  float
