"""
Numbers and figures for "The Shape of Evidence".

Every action is chosen by the weighted worst-case rule: an e-value profile
gives each action a certificate, max_z L(d,z)/E(z), and the rule takes the
action with the smallest one. The script reproduces

  Section 2.4 / 3.5   the two- and three-outcome examples and Table 1
  Section 3.3         the protection-information frontier (Figures 1, 2)
  Section 4           the shopkeeper application and Table 2 (Figure 3)
  Section 5           the controlled simulations (Figures 4, 5)

and writes the figures to results/figures/ and all reported numbers to
results/results.txt.
Everything is exact except the random-shape search of Section 4.4, which
uses a fixed seed.

Run:  python decision_pipeline.py
"""

import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle


# ---------------------------------------------------------------------------
# Paths and output
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
FIG = OUT / "figures"
MARKET_CSV = ROOT / "data" / "processed" / "polymarket_match_odds.csv"
FIG.mkdir(parents=True, exist_ok=True)

LINES = []


def log(s=""):
    print(s)
    LINES.append(s)


def header(title):
    log("\n" + "=" * 78)
    log(title)
    log("=" * 78)


# ---------------------------------------------------------------------------
# Figure style: serif text to match the thesis, drawn at the 15.2 cm text
# width so that LaTeX does not rescale the fonts.
# ---------------------------------------------------------------------------
COL = {
    "belief":  "#1F4E79",   # procedure targeting the belief's action
    "lr":      "#B03A2E",   # likelihood ratio
    "protect": "#1E6F5C",   # procedure targeting the protective action
    "flat":    "#7D6608",   # flat profile
    "ink":     "#1A1A1A",
    "ink2":    "#5A5A5A",
    "grid":    "#DDDDDD",
}
FIGW = 15.2 / 2.54

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "axes.titlesize": 10,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.titleweight": "normal",
    "axes.titlelocation": "left",
    "axes.titlepad": 7,
    "axes.edgecolor": "#999999",
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": COL["grid"],
    "grid.linewidth": 0.5,
    "grid.alpha": 0.9,
    "text.color": COL["ink"],
    "axes.labelcolor": COL["ink"],
    "xtick.color": COL["ink2"],
    "ytick.color": COL["ink2"],
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
    "legend.handlelength": 2.2,
    "legend.borderaxespad": 0.4,
    "lines.solid_capstyle": "round",
    "figure.dpi": 160,
    "savefig.dpi": 400,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "pdf.fonttype": 42,
})


def save_figure(fig, filename, *, layout=True):
    # Figures that set their own spacing pass layout=False.
    if layout:
        fig.set_constrained_layout(True)
    fig.savefig(FIG / filename, bbox_inches="tight", facecolor="white",
                metadata={"Creator": "decision_pipeline.py"})
    plt.close(fig)


def style_axes(ax, *, grid_axis="y"):
    ax.set_facecolor("white")
    ax.grid(True, axis=grid_axis, color=COL["grid"], linewidth=0.5)
    if grid_axis == "y":
        ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9A9A9A")
        ax.spines[side].set_linewidth(0.7)
    ax.tick_params(colors=COL["ink2"], length=3, width=0.7)


def end_label(ax, x, y, text, colour, *, dx=6, dy=0, va="center", size=8.2):
    """Label a line at its right end instead of in a legend."""
    ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                ha="left", va=va, fontsize=size, color=colour,
                annotation_clip=False)


# ---------------------------------------------------------------------------
# Certificates and the weighted worst-case rule
# ---------------------------------------------------------------------------
def certificate(L_d, e):
    """max_z L(d,z)/e(z), with x/0 = inf for x > 0 and 0/0 = 0."""
    ratio = np.where(L_d > 0,
                     np.divide(L_d, e, out=np.full_like(L_d, np.inf),
                               where=e > 0),
                     0.0)
    return ratio.max()


def rule(L, e):
    """Action with the smallest certificate (lowest index on ties).

    Certificates within rounding error of the minimum count as tied, so that
    exact ties such as the informative profiles are not broken by floating
    point.
    """
    certs = np.array([certificate(L[d], e) for d in range(len(L))])
    tied = np.flatnonzero(certs <= certs.min() * (1 + 1e-12))
    return int(tied[0]), certs


def expected_loss(P, L, d):
    return float(P @ L[d])


def likelihood_ratio(P0, P1):
    return P1 / P0


def flat(P0):
    return np.ones_like(P0)


def safe_profile(P0, L, d):
    """Theorem 3.1: scores proportional to the loss of d, divided by mu_d."""
    mu = P0 @ L[d]
    return L[d] / mu, mu


def protective_action(P0, L):
    return int(np.argmin(L @ P0))


def belief_action(P1, L):
    return int(np.argmin(L @ P1))


# The four rules compared in Section 5. Each returns (action, profile).
# The two procedures use the safe profile of their target, which selects
# that target by Corollary 3.2.
def procedure_belief(P0, P1, L):
    d = belief_action(P1, L)
    return d, safe_profile(P0, L, d)[0]


def procedure_protect(P0, P1, L):
    d = protective_action(P0, L)
    return d, safe_profile(P0, L, d)[0]


def rule_lr(P0, P1, L):
    e = likelihood_ratio(P0, P1)
    return rule(L, e)[0], e


def rule_flat(P0, P1, L):
    e = flat(P0)
    return rule(L, e)[0], e


RULES = {
    "belief":  (COL["belief"],  procedure_belief),
    "lr":      (COL["lr"],      rule_lr),
    "protect": (COL["protect"], procedure_protect),
    "flat":    (COL["flat"],    rule_flat),
}


# ---------------------------------------------------------------------------
# The examples of Section 2.4
# ---------------------------------------------------------------------------
L2 = np.array([[0., 8.],         # d1: back A
               [4., 0.]])        # d2: hedge
L3 = np.array([[0., 2., 10.],    # d_A
               [2., 0., 10.],    # d_B
               [4., 4., 0.]])    # d_C: insure against the draw
P0_2 = np.array([.5, .5])
P0_3 = np.ones(3) / 3


def informative_three_outcome():
    """Most informative profile leading to d_A in the three-outcome example.

    d_A ties with d_B whenever the draw term 10/e_C is the largest ratio, so
    the binding edge is the tie with d_C, 10/e_C = 4/e_B. Maximising
    0.6 log e_A + 0.4 log e_B + const subject to e_A + 3.5 e_B = 3 gives
    e_A = 9/5, e_B = 12/35 and e_C = 2.5 e_B = 6/7.
    """
    return np.array([9 / 5, 12 / 35, 6 / 7])


def check_three_outcome(e_star, P1, n=400):
    """Grid search over valid profiles leading to d_A, as a check."""
    best = -np.inf
    for a in np.linspace(1.5, 2.1, n):
        for b in np.linspace(0.25, 0.45, n):
            e = np.array([a, b, 3 - a - b])
            if e[2] <= 0 or rule(L3, e)[0] != 0:
                continue
            best = max(best, P1 @ np.log(e))
    return best <= P1 @ np.log(e_star) + 1e-9


def examples():
    header("Sections 2.4 and 3.5: the worked examples")

    P1 = np.array([.75, .25])
    rows = {
        "likelihood ratio (1.5, 0.5)": likelihood_ratio(P0_2, P1),
        "flat (1, 1)": flat(P0_2),
        "safe for d2 (2, 0)": safe_profile(P0_2, L2, 1)[0],
        "safe for d1 (0, 2)": safe_profile(P0_2, L2, 0)[0],
        "informative for d1 (2/3, 4/3)": np.array([2 / 3, 4 / 3]),
    }
    log("\nTwo outcomes: L(d1) = (0, 8), L(d2) = (4, 0), P0 = (1/2, 1/2), "
        "belief (0.75, 0.25)")
    log(f"{'profile':32s} {'C(d1)':>8s} {'C(d2)':>8s} {'action':>7s} "
        f"{'belief loss':>12s}")
    for name, e in rows.items():
        d, c = rule(L2, e)
        log(f"{name:32s} {c[0]:8.2f} {c[1]:8.2f} {'d' + str(d + 1):>7s} "
            f"{expected_loss(P1, L2, d):12.2f}")
    info = rows["informative for d1 (2/3, 4/3)"]
    log(f"log score of the informative profile: {P1 @ np.log(info):+.3f}")

    P1 = np.array([.6, .3, .1])
    e_star = informative_three_outcome()
    assert check_three_outcome(e_star, P1)
    hand = np.array([.3, .7, 2.0])
    rows3 = {
        "flat (1, 1, 1)": flat(P0_3),
        "likelihood ratio (1.8, 0.9, 0.3)": likelihood_ratio(P0_3, P1),
        "safe for d_C (1.5, 1.5, 0)": safe_profile(P0_3, L3, 2)[0],
        "safe for d_A (0, 0.5, 2.5)": safe_profile(P0_3, L3, 0)[0],
        "hand-picked (0.3, 0.7, 2.0)": hand,
        "informative for d_A (9/5, 12/35, 6/7)": e_star,
    }
    names = ["d_A", "d_B", "d_C"]
    log("\nThree outcomes: P0 uniform, belief (0.6, 0.3, 0.1)")
    log(f"{'profile':38s} {'C(d_A)':>8s} {'C(d_B)':>8s} {'C(d_C)':>8s} "
        f"{'action':>7s}")
    for name, e in rows3.items():
        d, c = rule(L3, e)
        log(f"{name:38s} {c[0]:8.2f} {c[1]:8.2f} {c[2]:8.2f} {names[d]:>7s}")
    log(f"log score, informative profile: {P1 @ np.log(e_star):+.3f}")
    log(f"log score, hand-picked profile: {P1 @ np.log(hand):+.3f}")


# ---------------------------------------------------------------------------
# Section 4: the shopkeeper and the Polymarket prices
# ---------------------------------------------------------------------------
# Outcomes, in order: Spain within ninety minutes (s), Spain later (s^e),
# Argentina within ninety minutes (a), Argentina later (a^e).
ORDERS = ["d1 mostly Spain", "d2 even split", "d3 mostly Argentina"]

# One hundred shirts split between the sides. A missing shirt costs the lost
# margin (3), an unsold one the markdown (1). The prices never enter the loss.
STOCK = 100
SHORT_COST, LEFT_COST = 3.0, 1.0
SPAIN_SHIRTS = [90, 50, 20]
DEMAND = [(80, 10), (100, 10), (10, 80), (10, 100)]   # (Spain, Argentina)


def shop_loss(spain_stock, z):
    want_s, want_a = DEMAND[z]
    arg_stock = STOCK - spain_stock
    short = SHORT_COST * (max(0, want_s - spain_stock)
                          + max(0, want_a - arg_stock))
    left = LEFT_COST * (max(0, spain_stock - want_s)
                        + max(0, arg_stock - want_a))
    return short + left


# In tens of units, as in the thesis.
L_SHOP = np.array([[shop_loss(a, z) for z in range(4)]
                   for a in SPAIN_SHIRTS]) / 10.0

# The model: three splits between the sides, each split again by the share of
# finals decided after ninety minutes (8 of 21 single-match finals, 1930-2022).
# The first split is Polymarket's tournament-winner price early in the group
# stage.
WINNER_PRICES = np.array([0.138, 0.116])
LATE_RATE = 8 / 21
SIDES = [tuple(WINNER_PRICES / WINNER_PRICES.sum()), (0.50, 0.50),
         (0.45, 0.55)]


def model(late_rate=LATE_RATE):
    return [np.array([s * (1 - late_rate), s * late_rate,
                      a * (1 - late_rate), a * late_rate])
            for s, a in SIDES]


def worst_member(P_list, loss_row):
    """Member of the model under which an action looks worst, and that loss."""
    means = [float(P @ loss_row) for P in P_list]
    i = int(np.argmax(means))
    return P_list[i], means[i]


def load_prices():
    with MARKET_CSV.open(newline="") as fh:
        return [{k: (v if k == "timestamp" else float(v)) for k, v in row.items()}
                for row in csv.DictReader(fh)]


def belief_from_prices(row):
    """Four outcomes from the result market and the trophy market."""
    return np.array([row["spain"], row["spain_advance"] - row["spain"],
                     row["argentina"], row["argentina_advance"] - row["argentina"]])


def capped_power(P_list, r, h, alpha):
    """Capped power shape, scaled by bisection so validity binds on the model."""
    x = r ** (1.0 / (1.0 - h))
    lo, hi = 1e-12, 1e12
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        e = np.minimum(x / mid, 1 / alpha)
        if max(float(P @ e) for P in P_list) > 1:
            lo = mid
        else:
            hi = mid
    return np.minimum(x / hi, 1 / alpha)


def capped_power_search(P_list, r, n_h=120, n_alpha=60):
    reached, best = set(), np.inf
    for h in np.linspace(-6, 0.95, n_h):
        for alpha in np.linspace(0.01, 0.95, n_alpha):
            e = capped_power(P_list, r, h, alpha)
            d = rule(L_SHOP, e)[0]
            reached.add(d)
            if d == 0:
                best = min(best, certificate(L_SHOP[0], e))
    return reached, best


def application(n_random=200_000, seed=0):
    header("Section 4: the shopkeeper's order")
    prices = load_prices()
    kickoff = prices[0]
    P1 = belief_from_prices(kickoff)
    P_model = model()
    P_star, _ = worst_member(P_model, L_SHOP[0])
    r = P1 / P_star

    result = [kickoff["spain"], kickoff["draw"], kickoff["argentina"]]
    log(f"{len(prices)} one-minute observations")
    log(f"kick-off result prices (Spain, draw, Argentina): {np.round(result, 4)}")
    log(f"kick-off trophy prices (Spain, Argentina): "
        f"{np.round([kickoff['spain_advance'], kickoff['argentina_advance']], 4)}")
    log(f"late wins {P1[1] + P1[3]:.4f} against draw price {result[1]:.4f}")
    log("\nmodel members:")
    for side, P in zip(SIDES, P_model):
        log(f"  sides {np.round(side, 4)} -> {np.round(P, 4)}")
    log(f"belief                 {np.round(P1, 4)}")
    log(f"worst member for d1    {np.round(P_star, 4)}")
    log(f"likelihood ratio r     {np.round(r, 4)}")

    mu = np.array([worst_member(P_model, L_SHOP[d])[1] for d in range(3)])
    belief_loss = L_SHOP @ P1
    log(f"\nworst-case expected loss {np.round(mu, 4)} -> {ORDERS[int(mu.argmin())]}")
    log(f"belief expected loss     {np.round(belief_loss, 4)} -> "
        f"{ORDERS[int(belief_loss.argmin())]}")

    safe1 = L_SHOP[0] / mu[0]
    lr_valid = r / max(P @ r for P in P_model)
    log(f"\nlikelihood ratio scaled down by {max(P @ r for P in P_model):.3f} "
        f"to be valid on the whole model")
    rows = {
        "likelihood ratio": lr_valid,
        "flat": flat(P_star),
        "safe for d2": L_SHOP[1] / mu[1],
        "safe for d1": safe1,
    }
    log(f"{'profile':18s} {'C1':>8s} {'C2':>8s} {'C3':>8s} {'action':>20s} "
        f"{'belief loss':>12s}")
    for name, e in rows.items():
        d, c = rule(L_SHOP, e)
        log(f"{name:18s} {c[0]:8.3f} {c[1]:8.3f} {c[2]:8.3f} {ORDERS[d]:>20s} "
            f"{expected_loss(P1, L_SHOP, d):12.3f}")
    log(f"safe profile for d1: ({', '.join(f'{v:.4f}' for v in safe1)}), "
        f"certificate {certificate(L_SHOP[0], safe1):.4f}")

    # Section 4.4: can an increasing reshaping of r reach d1?
    reached, best = capped_power_search(P_model, r)
    log(f"\ncapped power family, {120 * 60} (h, alpha) pairs")
    log(f"  orders reached: {sorted(ORDERS[d] for d in reached)}")
    log(f"  d1 reached: {'no' if best == np.inf else f'yes, certificate {best:.4f}'}")

    rng = np.random.default_rng(seed)
    order = np.argsort(r)
    reached, best = set(), np.inf
    for _ in range(n_random):
        e = np.empty(4)
        e[order] = np.sort(rng.random(4))
        e /= max(float(P @ e) for P in P_model)
        d = rule(L_SHOP, e)[0]
        reached.add(d)
        if d == 0:
            best = min(best, certificate(L_SHOP[0], e))
    log(f"{n_random} random increasing shapes")
    log(f"  orders reached: {sorted(ORDERS[d] for d in reached)}")
    log(f"  d1 reached: {'no' if best == np.inf else f'yes, certificate {best:.4f}'}")

    # The prediction-set rule of Section 1.1, for comparison.
    log("\nprediction set and worst case over the set:")
    by_belief = np.argsort(-P1)
    for alpha in (0.5, 0.3, 0.2, 0.1, 0.05):
        S, mass = [], 0.0
        for z in by_belief:
            S.append(int(z))
            mass += P1[z]
            if mass >= 1 - alpha:
                break
        worst = [max(L_SHOP[d][z] for z in S) for d in range(3)]
        d = int(np.argmin(worst))
        log(f"  coverage {1 - alpha:4.0%}  |S| = {len(S)}  -> {ORDERS[d]}")

    # Robustness to the share of finals decided late.
    log("\nhistorical late rate from 6/21 to 10/21:")
    n_disagree = n_unreached = 0
    for k in range(6, 11):
        P_k = model(k / 21)
        mu_k = np.array([worst_member(P_k, L_SHOP[d])[1] for d in range(3)])
        if int(mu_k.argmin()) == int(belief_loss.argmin()):
            continue
        n_disagree += 1
        r_k = P1 / worst_member(P_k, L_SHOP[0])[0]
        _, best_k = capped_power_search(P_k, r_k, n_h=60, n_alpha=30)
        n_unreached += best_k == np.inf
    log(f"  criteria disagree at {n_disagree} of 5 rates")
    log(f"  no capped power shape reaches d1 at {n_unreached} of those")


def figure_prices():
    prices = load_prices()
    t = [datetime.fromisoformat(r["timestamp"]) for r in prices]
    end_of_regulation, goal = 90, 162     # minutes of clock time after kick-off

    fig, (axT, axB) = plt.subplots(
        2, 1, figsize=(FIGW, 4.30), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.22})
    for ax in (axT, axB):
        style_axes(ax, grid_axis="y")
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(["0", "50", "100"])
        ax.set_ylabel("implied probability (%)", fontsize=8.2)
        ax.axvline(t[end_of_regulation], color="#C3C8CE", lw=0.8,
                   ls=(0, (3, 2)), zorder=1)
        ax.axvline(t[goal], color=COL["ink2"], lw=0.9, ls=(0, (1, 2)), zorder=1)

    for key, label, colour, dy in [("spain", "Spain", COL["belief"], -9),
                                   ("draw", "draw", COL["ink2"], 0),
                                   ("argentina", "Argentina", COL["lr"], 9)]:
        v = np.array([r[key] for r in prices])
        axT.plot(t, v, lw=1.9, color=colour, zorder=3)
        end_label(axT, t[-1], v[-1], label, colour, dx=7, dy=dy)

    spain = np.array([r["spain_advance"] for r in prices])
    axB.plot(t, spain, lw=1.9, color=COL["belief"], zorder=3)
    axB.plot(t, 1 - spain, lw=1.9, color=COL["lr"], zorder=3)
    end_label(axB, t[-1], spain[-1], "Spain", COL["belief"], dx=7)
    end_label(axB, t[-1], 1 - spain[-1], "Argentina", COL["lr"], dx=7)

    for i, text in [(45, "regulation"), (112, "extra time")]:
        axB.text(t[i], 0.06, text, ha="center", va="bottom", fontsize=7.6,
                 color="#9AA1A9", transform=axB.get_xaxis_transform())
    axB.annotate("Spain score in the $106$th minute", xy=(t[goal], 0.90),
                 xytext=(-12, 8), textcoords="offset points", ha="right",
                 va="bottom", fontsize=8.0, color=COL["ink2"],
                 arrowprops=dict(arrowstyle="-", color=COL["ink2"], lw=0.6,
                                 shrinkA=0, shrinkB=3))

    axT.set_title("Settles on the result after ninety minutes", loc="left",
                  pad=7, fontsize=9)
    axB.set_title("Settles on who lifts the trophy", loc="left", pad=7,
                  fontsize=9)
    axB.set_xlim(t[0], t[-1] + (t[-1] - t[0]) * 0.17)
    axB.xaxis.set_major_locator(mdates.HourLocator())
    axB.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axB.set_xlabel("time (UTC)")
    fig.subplots_adjust(left=0.09, right=0.80, bottom=0.12, top=0.92)
    save_figure(fig, "polymarket_match_odds.pdf")


# ---------------------------------------------------------------------------
# Section 3.3: geometry and the frontier
# ---------------------------------------------------------------------------
def figure_geometry():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(FIGW, 3.05))
    style_axes(axL, grid_axis="both")
    style_axes(axR, grid_axis="both")

    # (a) valid profiles, the budget line and the tie line e_B = 2 e_A
    tie = 1 / 3
    axL.fill([0, 2, 0], [0, 0, 2], color="#F0F0F0", zorder=0)
    axL.plot([0, 2], [2, 0], color=COL["ink2"], lw=1.0, zorder=2)
    axL.plot([0, 2 * tie], [2, 2 - 2 * tie], color=COL["belief"], lw=2.6,
             zorder=3, solid_capstyle="butt")
    axL.plot([2 * tie, 2], [2 - 2 * tie, 0], color=COL["lr"], lw=2.6,
             zorder=3, solid_capstyle="butt")
    xs = np.linspace(0, 2 / 3, 50)
    axL.plot(xs, 2 * xs, ls=(0, (4, 3)), color=COL["ink2"], lw=0.9, zorder=2)
    axL.text(0.245, 0.60, r"$e_B=2e_A$", fontsize=8, color=COL["ink2"],
             rotation=51, ha="center", va="bottom")

    points = {
        r"safe $d_1$": ((0.00, 2.00), (0.10, 0.06)),
        r"$\mathcal{E}^{\mathrm{info}}$": ((2 / 3, 4 / 3), (0.10, 0.08)),
        r"flat": ((1.00, 1.00), (0.09, 0.07)),
        r"LR": ((1.50, 0.50), (0.09, 0.07)),
        r"safe $d_2$": ((2.00, 0.00), (-0.05, 0.13)),
    }
    for label, ((x, y), (dx, dy)) in points.items():
        axL.scatter([x], [y], s=26, color=COL["ink"], zorder=5,
                    edgecolor="white", linewidth=0.8)
        axL.text(x + dx, y + dy, label, fontsize=8, color=COL["ink"])
    axL.text(0.05, 1.28, "selects $d_1$", fontsize=8, color=COL["belief"])
    axL.text(1.32, 0.95, "selects $d_2$", fontsize=8, color=COL["lr"])
    axL.set_xlim(-0.08, 2.30)
    axL.set_ylim(-0.10, 2.30)
    axL.set_xlabel(r"$e_A$")
    axL.set_ylabel(r"$e_B$")
    axL.set_xticks([0, 1, 2])
    axL.set_yticks([0, 1, 2])
    axL.set_aspect("equal")
    axL.set_title("(a) Valid profiles and the action boundary")

    # (b) both certificates along the budget line e = (2t, 2 - 2t)
    t = np.linspace(1e-3, 1 - 1e-3, 800)
    c1 = 8.0 / (2 - 2 * t)
    c2 = 4.0 / (2 * t)
    axR.plot(t, np.minimum(c1, c2), color=COL["ink"], lw=1.8, zorder=4)
    axR.plot(t, c1, ls=(0, (4, 3)), color=COL["belief"], lw=1.0, zorder=3)
    axR.plot(t, c2, ls=(0, (1.5, 2)), color=COL["lr"], lw=1.0, zorder=3)
    axR.axvline(tie, color=COL["ink2"], lw=0.7, ls=":")
    tf = t[t <= tie]
    axR.fill_between(tf, 0, np.minimum(8.0 / (2 - 2 * tf), 4.0 / (2 * tf)),
                     color=COL["belief"], alpha=0.05, lw=0, zorder=0)
    axR.scatter([1e-3, tie], [4.0, 6.0], s=26, color=COL["ink"], zorder=6,
                edgecolor="white", linewidth=0.8)
    axR.annotate(r"safe profile, $\mu_{d_1}=4$", xy=(0.0, 4.0),
                 xytext=(14, -26), textcoords="offset points",
                 fontsize=8, color=COL["ink2"], ha="left", va="top",
                 arrowprops=dict(arrowstyle="-", lw=0.6, color=COL["ink2"],
                                 shrinkA=0, shrinkB=3))
    axR.annotate(r"$\mathcal{E}^{\mathrm{info}}$, certificate $6$",
                 xy=(tie, 6.0), xytext=(16, 20), textcoords="offset points",
                 fontsize=8, color=COL["ink2"], ha="left", va="bottom",
                 arrowprops=dict(arrowstyle="-", lw=0.6, color=COL["ink2"],
                                 shrinkA=0, shrinkB=3))
    axR.text(0.90, 8.35, r"$C(d_1)$", fontsize=8, color=COL["belief"],
             ha="center", va="center")
    axR.text(0.10, 8.35, r"$C(d_2)$", fontsize=8, color=COL["lr"],
             ha="center", va="center")
    axR.text(tie / 2, 0.75, "the frontier", fontsize=8, color=COL["belief"],
             ha="center", va="center")
    axR.text(tie + 0.018, 0.75, r"$t=1/3$", fontsize=8, color=COL["ink2"],
             ha="left", va="center")
    axR.set_xlim(0, 1)
    axR.set_ylim(0, 9)
    axR.set_xlabel(r"position on the frontier, $e=(2t,\,2-2t)$")
    axR.set_ylabel("certificate")
    axR.set_title("(b) Certificates and the induced action")
    save_figure(fig, "toy_geometry.pdf")


def frontier():
    header("Section 3.3: the frontier for the bet d1")
    P1 = np.array([.75, .25])
    t = np.linspace(1e-4, 1 / 3, 900)
    eA, eB = 2 * t, 2 - 2 * t
    cert = 8.0 / eB
    score = P1[0] * np.log(eA) + P1[1] * np.log(eB)
    log(f"t -> 0:   profile (0, 2), certificate {cert[0]:.2f}, log score -> -inf")
    log(f"t = 1/3:  profile (2/3, 4/3), certificate {cert[-1]:.2f}, "
        f"log score {score[-1]:+.3f}")

    fig, ax = plt.subplots(figsize=(FIGW, 3.70))
    style_axes(ax, grid_axis="both")
    ax.fill_between(cert, score, score[-1], color=COL["belief"], alpha=0.045,
                    lw=0, zorder=0)
    ax.plot(cert, score, lw=2.3, color=COL["belief"], solid_capstyle="round",
            zorder=3)
    ax.scatter([cert[-1]], [score[-1]], s=46, color=COL["belief"], zorder=5,
               edgecolor="white", linewidth=1.3)
    ax.scatter([4.0], [score[0]], s=46, color=COL["protect"], zorder=5,
               edgecolor="white", linewidth=1.3)
    ax.annotate("protective endpoint\n" + r"$\mathcal{E}=(0,2)$",
                xy=(4.0, score[0]), xytext=(14, 8), textcoords="offset points",
                fontsize=8.2, color=COL["ink2"], ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=COL["ink2"], lw=0.6,
                                shrinkA=0, shrinkB=4))
    ax.annotate("informative endpoint\n" + r"$\mathcal{E}=(2/3,\,4/3)$",
                xy=(cert[-1], score[-1]), xytext=(-12, -26),
                textcoords="offset points", fontsize=8.2, color=COL["ink2"],
                ha="right", va="top",
                arrowprops=dict(arrowstyle="-", color=COL["ink2"], lw=0.6,
                                shrinkA=0, shrinkB=4))
    ax.text(5.42, -4.5, "profiles below the curve\nare dominated",
            fontsize=8.0, color=COL["ink2"], ha="center", va="center")
    ax.set_xlim(3.85, 6.45)
    ax.set_ylim(score[0] - 0.45, 0.75)
    ax.set_xlabel("certificate $C(d_1)$ (lower = better protection)")
    ax.set_ylabel("expected log-evidence (higher = more informative)")
    ax.text(0.5, 1.03, "every profile on this curve induces the same action $d_1$",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.4, color=COL["ink2"])
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.15, top=0.90)
    save_figure(fig, "sim_frontier.pdf")


# ---------------------------------------------------------------------------
# Section 5: controlled simulations
# ---------------------------------------------------------------------------
def belief_sweep():
    header("Section 5.1: varying the belief's confidence")
    p_grid = np.linspace(0.02, 0.98, 481)
    P_true = np.array([.75, .25])
    acts, loss = {}, {}
    for key, (_, choose) in RULES.items():
        a = np.array([choose(P0_2, np.array([p, 1 - p]), L2)[0]
                      for p in p_grid])
        acts[key] = a
        loss[key] = np.array([expected_loss(P_true, L2, d) for d in a])
    bets = {k: p_grid[acts[k] == 0] for k in RULES}
    log("belief's target bets for p >= 2/3; "
        f"first bet on the grid at p = {bets['belief'].min():.3f}")
    log("likelihood ratio bets for p <= 1/3; "
        f"last bet on the grid at p = {bets['lr'].max():.3f}")
    log(f"protective target and flat profile bet at "
        f"{len(bets['protect']) + len(bets['flat'])} grid points")

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(FIGW, 3.9), sharex=True,
        gridspec_kw={"height_ratios": [0.62, 1.25], "hspace": 0.30})
    style_axes(ax1, grid_axis="y")

    # Top: where each rule bets. Only two rules ever bet, so they get a lane
    # each and the other two are stated in the caption.
    ax0.set_facecolor("white")
    for side in ("top", "right", "left", "bottom"):
        ax0.spines[side].set_visible(False)
    ax0.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
    ax0.set_yticks([])
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0.16, 1.72)
    ax0.grid(False)
    ax0.axvspan(1 / 3, 2 / 3, color=COL["ink2"], alpha=0.055, lw=0, zorder=0)
    for x in (1 / 3, 2 / 3):
        ax0.axvline(x, color=COL["ink2"], lw=0.7, ls=(0, (1, 2)), zorder=1)

    lanes = [("belief", 1.14, "belief's target", r"bets when $p\geq 2/3$"),
             ("lr", 0.58, "likelihood ratio", r"bets when $p\leq 1/3$")]
    H = 0.30
    for key, y, name, when in lanes:
        a, colour = acts[key], RULES[key][0]
        ax0.plot([0, 1], [y, y], color="#C9CDD2", lw=1.0,
                 solid_capstyle="butt", zorder=2)
        change = np.flatnonzero(np.diff(a)) + 1
        for start, stop in zip(np.r_[0, change], np.r_[change, len(a)]):
            if a[start] != 0:
                continue
            x0 = p_grid[start]
            x1 = p_grid[stop] if stop < len(a) else p_grid[-1]
            ax0.add_patch(Rectangle((x0, y - H / 2), x1 - x0, H,
                                    facecolor=colour, edgecolor="none",
                                    zorder=3))
        ax0.text(-0.012, y, name, transform=ax0.get_yaxis_transform(),
                 ha="right", va="center", fontsize=8.2, color=COL["ink"])
        ax0.text(1.012, y, when, transform=ax0.get_yaxis_transform(),
                 ha="left", va="center", fontsize=7.6, color=colour)
    ax0.text(0.835, 1.33, "bets on $A$", fontsize=7.6, color=COL["belief"],
             ha="center", va="bottom")
    ax0.text(0.165, 0.77, "bets on $A$", fontsize=7.6, color=COL["lr"],
             ha="center", va="bottom")
    ax0.annotate("", xy=(1 / 3, 1.58), xytext=(2 / 3, 1.58),
                 arrowprops=dict(arrowstyle="<->", color=COL["ink2"], lw=0.7,
                                 shrinkA=0, shrinkB=0))
    ax0.text(0.5, 1.61, "neither bets", fontsize=7.6, color=COL["ink2"],
             ha="center", va="bottom")
    ax0.set_title("Where each rule bets on $A$ as confidence in $A$ rises",
                  loc="left", pad=6)

    # Bottom: what those actions cost under the truth.
    ax1.axvspan(1 / 3, 2 / 3, color=COL["ink2"], alpha=0.055, lw=0, zorder=0)
    for x in (1 / 3, 2 / 3):
        ax1.axvline(x, color=COL["ink2"], lw=0.7, ls=(0, (1, 2)), zorder=1)
    for key, ls in [("lr", (0, (5, 2))), ("belief", "-")]:
        ax1.plot(p_grid, loss[key], ls=ls, lw=2.0, color=RULES[key][0],
                 solid_capstyle="butt", zorder=3)
    end_label(ax1, p_grid[-1], loss["belief"][-1], "belief's target",
              COL["belief"])
    end_label(ax1, p_grid[-1], loss["lr"][-1], "likelihood ratio", COL["lr"])
    ax1.annotate("the likelihood ratio pays $3$\nwhere the belief's target pays $2$",
                 xy=(0.83, 2.97), xytext=(0.83, 2.60), ha="center", va="top",
                 fontsize=7.6, color=COL["ink2"],
                 arrowprops=dict(arrowstyle="-", color=COL["ink2"], lw=0.6,
                                 shrinkA=0, shrinkB=2))
    ax1.set_ylim(1.80, 3.30)
    ax1.set_yticks([2, 2.5, 3])
    ax1.set_xlim(0, 1)
    ax1.set_xticks([0, 1 / 3, 2 / 3, 1])
    ax1.set_xticklabels(["$0$", "$1/3$", "$2/3$", "$1$"])
    ax1.set_xlabel(r"belief $p=\widehat P_1[A]$   "
                   r"(truth: $P_{\mathrm{true}}[A]=0.75$)")
    ax1.set_ylabel("realised expected loss")
    ax1.set_title("What that action costs under the true distribution",
                  loc="left", pad=6)
    fig.subplots_adjust(left=0.175, right=0.815, bottom=0.135, top=0.925,
                        hspace=0.38)
    save_figure(fig, "sim_belief_sweep.pdf", layout=False)


def cost_ratio_sweep():
    header("Section 5.2: varying the cost ratio l/k")
    p = 0.75
    P1 = np.array([p, 1 - p])
    P_true = P1.copy()
    ratios = np.logspace(-1, 1, 600)
    curves = {k: [] for k in RULES}
    for x in ratios:
        L = np.array([[0., 4. * x], [4., 0.]])
        for key, (_, choose) in RULES.items():
            curves[key].append(expected_loss(P_true, L, choose(P0_2, P1, L)[0]))
    lo, hi = (1 - p) / p, p / (1 - p)
    log(f"the likelihood ratio takes the wrong action for {lo:.3f} < l/k < {hi:.1f}")
    # With the belief equal to the truth, its own target can never lose.
    assert all(np.all(np.array(curves["belief"]) <= np.array(curves[k]) + 1e-12)
               for k in RULES)

    fig, ax = plt.subplots(figsize=(FIGW, 3.05))
    style_axes(ax, grid_axis="y")
    ax.axvspan(lo, hi, color=COL["ink2"], alpha=0.06, lw=0, zorder=0)
    for x in (lo, hi):
        ax.axvline(x, color=COL["ink2"], lw=0.7, ls=(0, (1, 2)), zorder=1)
    for key, ls, z in [("protect", (0, (2, 2)), 2), ("lr", (0, (5, 2)), 3),
                       ("belief", "-", 4)]:
        ax.plot(ratios, curves[key], ls=ls, lw=1.9, color=RULES[key][0],
                zorder=z, solid_capstyle="round")
    ax.set_xscale("log")
    ax.set_xlim(ratios[0], ratios[-1] * 3.1)
    ax.set_ylim(-0.12, 3.72)

    ax.plot([2.0], [np.interp(2.0, ratios, curves["lr"])], "o", ms=4.0,
            color=COL["ink"], zorder=6)
    ax.text(0.40, 3.06, "likelihood ratio", fontsize=8.0, color=COL["lr"],
            ha="left", va="bottom")
    ax.text(0.96, 2.62, "protective action", fontsize=8.0,
            color=COL["protect"], ha="right", va="bottom")
    ax.text(2.05, 1.42, "belief's target", fontsize=8.0, color=COL["belief"],
            ha="left", va="center")
    ax.annotate("", xy=(lo, 3.46), xytext=(hi, 3.46),
                arrowprops=dict(arrowstyle="<->", color=COL["ink2"], lw=0.7,
                                shrinkA=0, shrinkB=0))
    ax.text(np.sqrt(lo * hi), 3.52,
            "the likelihood ratio induces the wrong action here",
            fontsize=7.8, color=COL["ink2"], ha="center", va="bottom")
    ax.annotate("the running example of Section 2",
                xy=(2.0, np.interp(2.0, ratios, curves["lr"])),
                xytext=(34, -30), textcoords="offset points", ha="left",
                va="center", fontsize=7.6, color=COL["ink2"],
                arrowprops=dict(arrowstyle="-", color=COL["ink2"], lw=0.6,
                                shrinkA=0, shrinkB=3))
    ax.set_xlabel("cost ratio $\\ell/k$ "
                  "(cost of a wrong bet relative to hedging)")
    ax.set_ylabel("realised expected loss")
    ax.set_title("What the likelihood ratio's failure window costs",
                 loc="left", pad=7)
    fig.subplots_adjust(left=0.105, right=0.775, bottom=0.185, top=0.885)
    save_figure(fig, "sim_asymmetry_window.pdf", layout=False)


def wrong_belief():
    header("Section 5.3: a belief that underrates the draw")
    P1 = np.array([.6, .3, .1])
    d_belief = procedure_belief(P0_3, P1, L3)[0]
    d_protect = procedure_protect(P0_3, P1, L3)[0]
    assert (d_belief, d_protect) == (0, 2)
    assert rule_lr(P0_3, P1, L3)[0] == 2 and rule_flat(P0_3, P1, L3)[0] == 2

    c = np.linspace(0.01, 0.45, 500)
    truth = np.stack([(1 - c) * 2 / 3, (1 - c) / 3, c], axis=1)
    gap = truth @ L3[d_belief] - truth @ L3[d_protect]
    log(f"belief's target d_A and protective action d_C break even at "
        f"c = {c[np.argmin(np.abs(gap))]:.3f} (exactly 1/4)")
    log("A stays the most likely outcome until c = 2/5")


if __name__ == "__main__":
    examples()
    application()
    figure_prices()
    figure_geometry()
    frontier()
    belief_sweep()
    cost_ratio_sweep()
    wrong_belief()
    (OUT / "results.txt").write_text("\n".join(LINES).lstrip("\n") + "\n")
    print(f"\nfigures -> {FIG}\nresults -> {OUT / 'results.txt'}")
