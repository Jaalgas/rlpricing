"""Stochastic 10 GB duopoly demand, adapted from Elliott et al. (2024).
Income bracket shares are from the supplied PDF, not independently verified.
Requires numpy and scipy. Prices in KZT; internal prices in EUR.
"""
import numpy as np
from scipy.special import exp1, expit

class CustomerSimulator:
    def __init__(self, exchange_rate, arrival_rate=100, seed=None,
                 allow_no_purchase=True, upper_income=1000000,
                 speed_a=0.01113875, speed_b=0.01113875):
        if exchange_rate <= 0 or arrival_rate < 0 or upper_income <= 500000 or min(speed_a, speed_b) <= 0:
            raise ValueError('Invalid market parameters')
        self.rng = np.random.default_rng(seed)
        self.exchange_rate = exchange_rate
        self.arrival_rate = arrival_rate
        self.allow_no_purchase = allow_no_purchase
        self.speeds = np.array([speed_a, speed_b])
        self.cap = 10.0
        self.theta_c = 1.405e-4
        self.sigma = 0.683
        self.bounds = np.array([0,40000,70000,100000,150000,300000,500000,upper_income], dtype=float)
        shares = np.array([5.1,3.5,6.4,12.9,30.4,23.4,13.8])
        self.shares = shares / shares.sum()

    def generate_customers(self, duration):
        if duration < 0:
            raise ValueError('Duration must be nonnegative')
        n = self.rng.poisson(self.arrival_rate * duration)
        bracket = self.rng.choice(7, size=n, p=self.shares)
        income = self.rng.uniform(self.bounds[bracket], self.bounds[bracket+1])
        z = 12 * income / (10000 * self.exchange_rate)
        theta_p = np.exp(-1.859 - 0.727*z)
        theta_d = np.exp(0.597 + 0.335*z)
        return income, theta_p, theta_d

    def expected_data_utility(self, theta_d):
        # Exact expectation of optimized utility under a hard data cap.
        d = np.asarray(theta_d)[:, None]
        a = self.theta_c / self.speeds[None, :]
        return (exp1(d*a) - exp1(d*a*(1+self.cap))) / d

    def choice_probabilities(self, theta_p, theta_d, price_a, price_b):
        prices = np.array([price_a, price_b]) / self.exchange_rate
        V = self.expected_data_utility(theta_d) - theta_p[:, None]*prices
        tau = 1-self.sigma
        scaled = V/tau
        logsum = np.logaddexp(scaled[:,0], scaled[:,1])
        conditional = np.exp(scaled-logsum[:,None])
        buy = expit(tau*logsum) if self.allow_no_purchase else np.ones(len(theta_p))
        return np.column_stack((conditional*buy[:,None], 1-buy))

    def __call__(self, duration, price_a, price_b):
        income, theta_p, theta_d = self.generate_customers(duration)
        probs = self.choice_probabilities(theta_p, theta_d, price_a, price_b)
        u = self.rng.random(len(income))
        choice = np.where(u < probs[:,0], 0, np.where(u < probs[:,0]+probs[:,1], 1, 2))
        # Optional realized consumption, sampled only after choice.
        valuation = self.rng.exponential(scale=1/theta_d)
        selected_speed = self.speeds[np.minimum(choice,1)]
        consumption = np.clip(valuation*selected_speed/self.theta_c-1, 0, self.cap)
        consumption[choice==2] = 0
        self.last_batch = dict(income=income, theta_p=theta_p, theta_d=theta_d,
                               choice=choice, consumption_gb=consumption, probabilities=probs)
        return tuple(map(int, np.bincount(choice, minlength=3)))

if __name__ == '__main__':
    # Example exchange rate only, not a verified current FX quote.
    sim = CustomerSimulator(exchange_rate=600, seed=42)
    for t in range(5):
        print(t+1, sim(1,2500,2800))