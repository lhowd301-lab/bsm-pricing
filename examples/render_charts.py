"""Export the existing visualization functions without a desktop backend."""
from pathlib import Path
import sys
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import bsm_visualization as charts

random.seed(42)
np.random.seed(42)
out = root / "figures"
out.mkdir(exist_ok=True)
functions = [
    ("01_payoff_price", charts.plot_payoff_and_price),
    ("02_greeks", charts.plot_greeks),
    ("03_price_paths", charts.plot_price_paths),
    ("04_vol_time_surface", charts.plot_vol_time_surface),
    ("05_volatility_smile", charts.plot_volatility_smile),
    ("06_time_decay", charts.plot_time_decay),
]
for name, function in functions:
    figure = function()
    figure.savefig(out / (name + ".png"), dpi=150, bbox_inches="tight")
    charts.plt.close(figure)
    print(name, "OK")
