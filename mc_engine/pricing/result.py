from dataclasses import dataclass, field

@dataclass
class PricingResult:
    price:     float
    std_error: float
    ci_lower:  float
    ci_upper:  float