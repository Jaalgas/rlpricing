class TwoBound:
    def __init__(self, min_p, max_p, diff, hard_upper_limit=False):
        self.min = min_p
        self.max = max_p
        self.diff = diff
        self.hard_upper_limit = hard_upper_limit

    def __call__(self, ref_p, conc_p):
        target = conc_p - self.diff
        if self.hard_upper_limit:
            target = min(target, self.max)
        return self.max if target < self.min else target


class CartelTwoBound:
    def __init__(self, min_p, max_p, diff, cartel_p, hard_upper_limit=False):
        self.two_bound = TwoBound(min_p, max_p, diff, hard_upper_limit)
        self.min = min_p
        self.max = max_p
        self.cartel_p = cartel_p
        self.diff = diff

    def __call__(self, ref_p, conc_p):
        if round(conc_p) == round(self.cartel_p):
            return conc_p
        else:
            return self.two_bound(ref_p, conc_p)
