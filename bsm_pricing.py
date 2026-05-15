import math
import random


def norm_cdf(x):
    """Standard normal CDF"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x):
    """Standard normal PDF"""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bsm_price(S, K, T, r, sigma, option_type='call'):
    """
    Black-Scholes-Merton option pricing model.

    Parameters:
        S: spot price of the underlying
        K: strike price
        T: time to maturity (years)
        r: risk-free rate
        sigma: annualized volatility
        option_type: 'call' or 'put'
    Returns:
        theoretical option price
    """
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == 'call':
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def bsm_greeks(S, K, T, r, sigma, option_type='call'):
    """Compute all Greeks."""
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    pdf_d1 = norm_pdf(d1)

    if option_type == 'call':
        delta = norm_cdf(d1)
        theta = (-S * pdf_d1 * sigma / (2.0 * math.sqrt(T))
                 - r * K * math.exp(-r * T) * norm_cdf(d2))
        rho_mult = norm_cdf(d2)
    else:
        delta = norm_cdf(d1) - 1.0
        theta = (-S * pdf_d1 * sigma / (2.0 * math.sqrt(T))
                 + r * K * math.exp(-r * T) * norm_cdf(-d2))
        rho_mult = -norm_cdf(-d2)

    gamma = pdf_d1 / (S * sigma * math.sqrt(T))
    vega = S * pdf_d1 * math.sqrt(T) / 100.0   # per 1% vol change
    theta_daily = theta / 365.0                  # per day
    rho = K * T * math.exp(-r * T) * rho_mult / 100.0  # per 1% rate change

    return {
        'delta': round(delta, 4),
        'gamma': round(gamma, 4),
        'theta': round(theta_daily, 4),
        'vega':  round(vega, 4),
        'rho':   round(rho, 4),
    }


def simulate_price_paths(S0, r, sigma, T, n_steps=252, n_paths=100, seed=42):
    """
    Monte Carlo simulation of geometric Brownian motion price paths.
    Returns list of paths, each path is a list of prices [S0, S1, ..., S_nsteps].
    """
    random.seed(seed)
    dt = T / n_steps
    drift = (r - 0.5 * sigma ** 2) * dt
    shock = sigma * math.sqrt(dt)

    paths = [[S0] for _ in range(n_paths)]

    for step in range(n_steps):
        for p in range(n_paths):
            Z = random.gauss(0.0, 1.0)
            prev = paths[p][-1]
            paths[p].append(prev * math.exp(drift + shock * Z))

    return paths


def mc_option_price(S0, K, T, r, sigma, option_type='call',
                    n_paths=100000, seed=42):
    """Vanilla Monte Carlo option pricing. Returns (price, std_err)."""
    random.seed(seed)
    drift = (r - 0.5 * sigma ** 2) * T
    shock = sigma * math.sqrt(T)

    payoffs = []
    for _ in range(n_paths):
        ST = S0 * math.exp(drift + shock * random.gauss(0.0, 1.0))
        payoff = max(ST - K, 0.0) if option_type == 'call' else max(K - ST, 0.0)
        payoffs.append(payoff)

    n = len(payoffs)
    mean_p = sum(payoffs) / n
    variance = sum((x - mean_p) ** 2 for x in payoffs) / (n - 1)
    std_err = math.sqrt(variance / n)

    return math.exp(-r * T) * mean_p, math.exp(-r * T) * std_err


def print_price_table(S0, K_list, T, r, sigma):
    """Print option prices for different strikes."""
    header = "| Strike |  Call Price  |  Put Price  |"
    sep =    "+--------+--------------+-------------+"
    print("\n  " + sep)
    print("  " + header)
    print("  " + sep)
    for K in K_list:
        c = bsm_price(S0, K, T, r, sigma, 'call')
        p = bsm_price(S0, K, T, r, sigma, 'put')
        print(f"  | {K:6.0f} |    {c:8.4f}    |   {p:8.4f}   |")
    print("  " + sep)


# ── Demo ──
if __name__ == '__main__':
    S0    = 100.0   # spot price
    K     = 100.0   # strike price
    T     = 1.0     # 1 year
    r     = 0.03    # 3% risk-free rate
    sigma = 0.25    # 25% volatility

    print("=" * 60)
    print("  Black-Scholes-Merton Option Pricing Model")
    print("  (Pure Python, zero dependencies)")
    print("=" * 60)
    print(f"  Spot       S0   = {S0:.0f}")
    print(f"  Strike     K    = {K:.0f}")
    print(f"  Maturity   T    = {T:.1f} yr")
    print(f"  Risk-free  r    = {r:.2%}")
    print(f"  Volatility sigma = {sigma:.2%}")
    print("-" * 60)

    call = bsm_price(S0, K, T, r, sigma, 'call')
    put  = bsm_price(S0, K, T, r, sigma, 'put')

    print(f"\n  Call price : {call:.4f}")
    print(f"  Put  price : {put:.4f}")

    print("\n  Put-Call Parity: C - P = S - K*exp(-rT)")
    lhs = call - put
    rhs = S0 - K * math.exp(-r * T)
    print(f"  LHS = {lhs:.6f}")
    print(f"  RHS = {rhs:.6f}")
    print(f"  Diff = {abs(lhs - rhs):.2e}  [OK]" if abs(lhs - rhs) < 1e-10
          else f"  WARNING: parity violation {abs(lhs-rhs):.4f}")

    print("\n  Greeks:")
    for ot in ['call', 'put']:
        g = bsm_greeks(S0, K, T, r, sigma, ot)
        label = "Call" if ot == 'call' else "Put "
        print(f"  {label}: delta={g['delta']:+.4f}  gamma={g['gamma']:.4f}  "
              f"theta={g['theta']:+.4f}/day  vega={g['vega']:.4f}  "
              f"rho={g['rho']:+.4f}")

    print("\n  Monte Carlo verification (100k paths):")
    mc_c, se_c = mc_option_price(S0, K, T, r, sigma, 'call')
    mc_p, se_p = mc_option_price(S0, K, T, r, sigma, 'put')
    print(f"  Call MC: {mc_c:.4f} +/- {1.96*se_c:.4f} (95% CI)")
    print(f"  Put  MC: {mc_p:.4f} +/- {1.96*se_p:.4f} (95% CI)")
    print(f"  Call MC error: {abs(mc_c - call):.4f}")
    print(f"  Put  MC error: {abs(mc_p - put):.4f}")

    print_price_table(S0, range(80, 121, 10), T, r, sigma)

    print("\n  Simulated price paths (5 paths, 252 steps):")
    paths = simulate_price_paths(S0, r, sigma, T, n_paths=5)
    for i, p in enumerate(paths):
        final = p[-1]
        arrow = "UP" if final > S0 else "DN"
        print(f"  Path {i+1}: final={final:.2f} ({arrow}), change={final-S0:+.2f}")

    print("\n" + "=" * 60)
