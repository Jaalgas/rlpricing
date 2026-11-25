from .static import CartelTwoBound, TwoBound


class Fixed:
    """The competitor offers the same KZT price throughout the run."""

    def __init__(self, p):
        self.price = float(p)

    def __call__(self, ref_p, conc_p):
        return self.price


COMPETITORS = {"fixed": Fixed, "twobound": TwoBound, "cartel": CartelTwoBound}
