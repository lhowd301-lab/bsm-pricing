"""
BSM Option Pricing — Comprehensive Visualization
Requires: pip install matplotlib numpy
"""
import math
import random
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from bsm_pricing import (bsm_price, bsm_greeks, simulate_price_paths,
                         mc_option_price)

# ── Global parameters ──
S0 = 100.0
K = 100.0
T = 1.0
r = 0.03
sigma = 0.25

plt.rcParams.update({
    'figure.dpi': 120,
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})


# ═══════════════════════════════════════════════
# Fig 1: Option Payoff & Price Curves
# ═══════════════════════════════════════════════
def plot_payoff_and_price():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    spot_range = np.linspace(60, 140, 200)
    times = [0.01, 0.25, 0.5, 0.75, 1.0]

    # -- Left: Call --
    for t in times:
        prices = [bsm_price(s, K, t, r, sigma, 'call') for s in spot_range]
        ax1.plot(spot_range, prices, lw=1.5, label=f'T={t:.2f}')
    payoff = np.maximum(spot_range - K, 0)
    ax1.plot(spot_range, payoff, 'k--', lw=1.8, label='Payoff at expiry')
    ax1.axvline(K, color='gray', ls=':', alpha=0.6)
    ax1.set_title('Call Option Price vs Spot')
    ax1.set_xlabel('Spot Price S')
    ax1.set_ylabel('Option Price')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # -- Right: Put --
    for t in times:
        prices = [bsm_price(s, K, t, r, sigma, 'put') for s in spot_range]
        ax2.plot(spot_range, prices, lw=1.5, label=f'T={t:.2f}')
    payoff = np.maximum(K - spot_range, 0)
    ax2.plot(spot_range, payoff, 'k--', lw=1.8, label='Payoff at expiry')
    ax2.axvline(K, color='gray', ls=':', alpha=0.6)
    ax2.set_title('Put Option Price vs Spot')
    ax2.set_xlabel('Spot Price S')
    ax2.set_ylabel('Option Price')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'BSM Option Pricing  (K={K}, r={r:.0%}, sigma={sigma:.0%})',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════
# Fig 2: Greeks Panorama
# ═══════════════════════════════════════════════
def plot_greeks():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    spot_range = np.linspace(50, 150, 300)
    greek_names = ['delta', 'gamma', 'theta', 'vega', 'rho']

    for i, gname in enumerate(greek_names):
        ax = axes.flat[i]
        for ot, color, ls in [('call', '#2196F3', '-'), ('put', '#F44336', '--')]:
            vals = [bsm_greeks(s, K, T, r, sigma, ot)[gname] for s in spot_range]
            ax.plot(spot_range, vals, color=color, ls=ls, lw=1.8, label=f'{ot}')
        ax.axvline(K, color='gray', ls=':', alpha=0.5)
        ax.set_title(gname.upper())
        ax.set_xlabel('Spot Price S')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Bottom-right: summary table
    ax_table = axes.flat[5]
    ax_table.axis('off')
    g_call = bsm_greeks(S0, K, T, r, sigma, 'call')
    g_put = bsm_greeks(S0, K, T, r, sigma, 'put')
    rows = ['delta', 'gamma', 'theta', 'vega', 'rho']
    col_labels = ['Greek', 'Call', 'Put']
    cell_text = [[r.upper(), f"{g_call[r]:+.4f}", f"{g_put[r]:+.4f}"] for r in rows]
    table = ax_table.table(cellText=cell_text, colLabels=col_labels,
                           cellLoc='center', loc='center',
                           colColours=['#E0E0E0']*3)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax_table.set_title(f'Current Greeks  (S={S0}, K={K})', fontsize=12, fontweight='bold', y=0.85)

    fig.suptitle('BSM Greeks Panorama', fontsize=15, fontweight='bold')
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════
# Fig 3: Monte Carlo Price Paths
# ═══════════════════════════════════════════════
def plot_price_paths():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # -- Left: sample paths --
    n_paths = 30
    paths = simulate_price_paths(S0, r, sigma, T, n_steps=252, n_paths=n_paths)
    time_axis = np.linspace(0, T, 253)
    for i, p in enumerate(paths):
        alpha_val = 0.6 if i < 5 else 0.15
        lw_val = 1.2 if i < 5 else 0.5
        ax1.plot(time_axis, p, lw=lw_val, alpha=alpha_val,
                 color='#1976D2' if p[-1] >= S0 else '#D32F2F')
    ax1.axhline(S0, color='black', ls='--', lw=1.2, alpha=0.7, label=f'S0={S0}')
    ax1.axhline(K, color='gray', ls=':', lw=1, alpha=0.7, label=f'K={K}')
    ax1.set_title(f'{n_paths} Simulated Price Paths (GBM)')
    ax1.set_xlabel('Time (years)')
    ax1.set_ylabel('Price')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # -- Right: terminal distribution histogram --
    n_mc = 50000
    random.seed(42)
    drift = (r - 0.5 * sigma ** 2) * T
    shock = sigma * math.sqrt(T)
    finals = [S0 * math.exp(drift + shock * random.gauss(0, 1)) for _ in range(n_mc)]
    mean_final = np.mean(finals)

    ax2.hist(finals, bins=80, density=True, alpha=0.7, color='#42A5F5',
             edgecolor='white', linewidth=0.3)
    ax2.axvline(S0, color='black', ls='--', lw=1.5, label=f'S0={S0}')
    ax2.axvline(K, color='gray', ls=':', lw=1.5, label=f'K={K}')
    ax2.axvline(mean_final, color='#D32F2F', ls='-', lw=1.5,
                label=f'E[ST]={mean_final:.2f}')

    # overlay theoretical lognormal PDF
    x_vals = np.linspace(min(finals), max(finals), 300)
    lognorm_mean = math.log(S0) + (r - 0.5*sigma**2) * T
    lognorm_std = sigma * math.sqrt(T)
    pdf_vals = (np.exp(-0.5 * ((np.log(x_vals/S0) - (r - 0.5*sigma**2)*T)
                / (sigma*np.sqrt(T)))**2)
                / (x_vals * sigma * np.sqrt(2*np.pi*T)))
    ax2.plot(x_vals, pdf_vals, 'r-', lw=1.5, alpha=0.8, label='Theoretical (log-normal)')

    ax2.set_title(f'Terminal Price Distribution  (n={n_mc:,})')
    ax2.set_xlabel('Terminal Price ST')
    ax2.set_ylabel('Probability Density')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'Monte Carlo Simulation  (S0={S0}, sigma={sigma:.0%}, r={r:.0%}, T={T}yr)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════
# Fig 4: Volatility & Time Price Surface (3D/Heatmap)
# ═══════════════════════════════════════════════
def plot_vol_time_surface():
    fig = plt.figure(figsize=(16, 6))

    vol_range = np.linspace(0.05, 0.80, 60)
    time_range = np.linspace(0.05, 2.0, 60)
    VV, TT = np.meshgrid(vol_range, time_range)
    call_prices = np.zeros_like(VV)
    put_prices = np.zeros_like(VV)

    for i in range(len(time_range)):
        for j in range(len(vol_range)):
            call_prices[i, j] = bsm_price(S0, K, time_range[i], r, vol_range[j], 'call')
            put_prices[i, j] = bsm_price(S0, K, time_range[i], r, vol_range[j], 'put')

    # Call heatmap
    ax1 = fig.add_subplot(1, 3, 1)
    c1 = ax1.contourf(TT, VV, call_prices, levels=30, cmap='RdYlBu_r')
    ax1.set_title('Call Price')
    ax1.set_xlabel('Time to Maturity T')
    ax1.set_ylabel('Volatility sigma')
    plt.colorbar(c1, ax=ax1, shrink=0.85)

    # Put heatmap
    ax2 = fig.add_subplot(1, 3, 2)
    c2 = ax2.contourf(TT, VV, put_prices, levels=30, cmap='RdYlBu_r')
    ax2.set_title('Put Price')
    ax2.set_xlabel('Time to Maturity T')
    ax2.set_ylabel('Volatility sigma')
    plt.colorbar(c2, ax=ax2, shrink=0.85)

    # 3D surface
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.plot_surface(TT, VV, call_prices, cmap='coolwarm', alpha=0.85,
                     linewidth=0, antialiased=True)
    ax3.set_title('Call Price Surface')
    ax3.set_xlabel('T')
    ax3.set_ylabel('sigma')
    ax3.set_zlabel('Price')
    ax3.view_init(elev=25, azim=-60)

    fig.suptitle(f'Volatility-Time Price Surface  (S={S0}, K={K}, r={r:.0%})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════
# Fig 5: Volatility Smile (synthetic)
# ═══════════════════════════════════════════════
def plot_volatility_smile():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    strike_range = np.linspace(70, 130, 31)
    atm_vol = 0.25

    # build smile: OTM puts & ITM calls carry a premium
    def smile_vol(K_i):
        moneyness = K_i / S0
        return atm_vol + 0.08 * (moneyness - 1)**2 + 0.03 * abs(moneyness - 1)

    smile_vols = [smile_vol(k) for k in strike_range]

    ax.plot(strike_range, [s*100 for s in smile_vols], 'ko-', ms=4, lw=1.5,
            label='Implied Volatility (smile)')
    ax.axhline(atm_vol*100, color='gray', ls='--', lw=1, alpha=0.7,
               label=f'ATM sigma={atm_vol:.0%}')
    ax.axvline(S0, color='gray', ls=':', alpha=0.5)
    ax.fill_between(strike_range, atm_vol*100, [s*100 for s in smile_vols],
                    alpha=0.15, color='#FF9800')
    ax.set_xlabel('Strike K')
    ax.set_ylabel('Implied Volatility (%)')
    ax.set_title('Volatility Smile (Synthetic)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════
# Fig 6: Greeks & Price Time Decay
# ═══════════════════════════════════════════════
def plot_time_decay():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    time_range = np.linspace(0.02, 2.0, 200)
    greek_names = ['delta', 'gamma', 'theta', 'vega', 'rho', 'price']

    for i, gname in enumerate(greek_names):
        ax = axes.flat[i]
        for ot, color, ls in [('call', '#2196F3', '-'), ('put', '#F44336', '--')]:
            if gname == 'price':
                vals = [bsm_price(S0, K, t, r, sigma, ot) for t in time_range]
            else:
                vals = [bsm_greeks(S0, K, t, r, sigma, ot)[gname] for t in time_range]
            ax.plot(time_range, vals, color=color, ls=ls, lw=1.8, label=ot)
        ax.set_title(gname.upper())
        ax.set_xlabel('Time to Maturity T (years)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Greeks & Price Time Decay  (S={S0}, K={K}, sigma={sigma:.0%})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════
if __name__ == '__main__':
    print("Generating BSM visualization charts...")
    print()

    figures = []

    print("  [1/6] Option Payoff & Price Curves...")
    figures.append(plot_payoff_and_price())

    print("  [2/6] Greeks Panorama...")
    figures.append(plot_greeks())

    print("  [3/6] Monte Carlo Price Paths...")
    figures.append(plot_price_paths())

    print("  [4/6] Volatility-Time Price Surface...")
    figures.append(plot_vol_time_surface())

    print("  [5/6] Volatility Smile...")
    figures.append(plot_volatility_smile())

    print("  [6/6] Greeks Time Decay...")
    figures.append(plot_time_decay())

    print(f"\nDone! {len(figures)} charts generated. Displaying now...")
    plt.show()
