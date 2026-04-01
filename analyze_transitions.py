"""
Комплексный анализ свойств SDE-траекторий при разных параметрах.

Проверяемые гипотезы:
  H1: Перелёты через границу pi ~ half-normal (ЦПТ)
  H2: Характер распределения временных расстояний между CP
  H3: Число CP на ряд ~ Poisson (стационарность процесса)
  H4: Holding times ~ exponential (теория Крамерса)

alpha = sqrt(D).

Результаты:
  figures/hypothesis_summary.pdf    — сводка всех гипотез
  figures/overshoot_hist_examples.pdf
  figures/overshoot_halfnorm_heatmap.pdf
  figures/mean_cp_heatmap.pdf
  results/transition_analysis/summary.csv
"""

import sys
import os
import itertools
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from algoritms.data_utils import generate_sde_trajectory, compute_levels

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})


# ======================================================================
# Metrics
# ======================================================================

def compute_inter_cp_distances(cp_labels):
    cp_idx = np.where(cp_labels == 1)[0]
    if len(cp_idx) < 2:
        return np.array([])
    return np.diff(cp_idx)


def compute_boundary_overshoots(x_values, levels):
    overshoots = []
    for i in range(1, len(levels)):
        if levels[i] != levels[i - 1]:
            nearest = round(x_values[i] / math.pi) * math.pi
            overshoots.append(abs(x_values[i] - nearest))
    return np.array(overshoots)


def compute_holding_times(levels):
    times = []
    c = 0
    for i in range(len(levels) - 1):
        if levels[i] != levels[i + 1]:
            times.append(c)
            c = 0
        else:
            c += 1
    return np.array(times)


# ======================================================================
# Tests
# ======================================================================

def chi2_gof(data, dist_name, n_bins=12):
    if len(data) < 20:
        return np.nan
    d = data.astype(float)
    n = len(d)
    edges = np.linspace(max(d.min() - 0.5, 0), np.percentile(d, 98), n_bins + 1)
    obs, _ = np.histogram(d, bins=edges)
    exp = np.zeros(n_bins)

    if dist_name == "halfnorm":
        sigma = np.sqrt(np.mean(d ** 2))
        for j in range(n_bins):
            exp[j] = (stats.halfnorm.cdf(edges[j+1], scale=sigma)
                      - stats.halfnorm.cdf(edges[j], scale=sigma)) * n
    elif dist_name == "expon":
        for j in range(n_bins):
            exp[j] = (stats.expon.cdf(edges[j+1], scale=d.mean())
                      - stats.expon.cdf(edges[j], scale=d.mean())) * n
    elif dist_name == "gamma":
        try:
            a, _, sc = stats.gamma.fit(d, floc=0)
            for j in range(n_bins):
                exp[j] = (stats.gamma.cdf(edges[j+1], a, 0, sc)
                          - stats.gamma.cdf(edges[j], a, 0, sc)) * n
        except Exception:
            return np.nan
    elif dist_name == "beta":
        try:
            d_scaled = d / (d.max() + 1e-9)
            a_b, b_b, loc_b, sc_b = stats.beta.fit(d_scaled)
            for j in range(n_bins):
                lo = edges[j] / (d.max() + 1e-9)
                hi = edges[j+1] / (d.max() + 1e-9)
                exp[j] = (stats.beta.cdf(hi, a_b, b_b, loc_b, sc_b)
                          - stats.beta.cdf(lo, a_b, b_b, loc_b, sc_b)) * n
        except Exception:
            return np.nan

    mask = exp >= 3
    if mask.sum() < 3:
        return np.nan
    o, e = obs[mask].astype(float), exp[mask]
    e = e * o.sum() / e.sum()
    _, p = stats.chisquare(o, f_exp=e)
    return p


def poisson_dispersion_test(cp_counts):
    n = len(cp_counts)
    if n < 5:
        return np.nan, np.nan
    mean_c = np.mean(cp_counts)
    if mean_c < 0.5:
        return np.nan, np.nan
    var_c = np.var(cp_counts, ddof=1)
    dispersion_index = var_c / mean_c
    chi2_stat = (n - 1) * var_c / mean_c
    p_val = 2 * min(stats.chi2.cdf(chi2_stat, df=n-1),
                    1 - stats.chi2.cdf(chi2_stat, df=n-1))
    return dispersion_index, p_val


# ======================================================================
# Grid
# ======================================================================

def run_grid(D_values, dt_values, noise_types,
             length=10000, n_series=30, base_seed=0):
    records = []
    all_overshoots = {}
    all_time_dists = {}

    total = len(D_values) * len(dt_values) * len(noise_types)
    done = 0

    for D, dt_val, noise in itertools.product(D_values, dt_values, noise_types):
        alpha = math.sqrt(D)
        ov_list, td_list, ht_list, cp_counts = [], [], [], []

        for i in range(n_series):
            seed = base_seed + i
            x = generate_sde_trajectory(length, dt=dt_val, D=D,
                                        noise_type=noise, seed=seed)
            levels, cp_labels = compute_levels(x)
            ov_list.append(compute_boundary_overshoots(x, levels))
            td_list.append(compute_inter_cp_distances(cp_labels))
            ht_list.append(compute_holding_times(levels))
            cp_counts.append(int(cp_labels.sum()))

        all_ov = np.concatenate(ov_list) if ov_list else np.array([])
        all_td = np.concatenate(td_list) if td_list else np.array([])
        all_ht = np.concatenate(ht_list) if ht_list else np.array([])
        pos_td = all_td[all_td > 0].astype(float) if len(all_td) > 0 else np.array([])
        pos_ht = all_ht[all_ht > 0].astype(float) if len(all_ht) > 0 else np.array([])

        disp_idx, disp_p = poisson_dispersion_test(np.array(cp_counts))

        # Gamma shape parameter for time distances
        gamma_shape_td = np.nan
        if len(pos_td) >= 20:
            try:
                gamma_shape_td, _, _ = stats.gamma.fit(pos_td, floc=0)
            except Exception:
                pass

        gamma_shape_ht = np.nan
        if len(pos_ht) >= 20:
            try:
                gamma_shape_ht, _, _ = stats.gamma.fit(pos_ht, floc=0)
            except Exception:
                pass

        rec = {
            "D": D, "alpha": round(alpha, 4), "dt": dt_val,
            "noise_type": noise,
            "mean_cp": np.mean(cp_counts),
            "std_cp": np.std(cp_counts),
            # H1
            "h1_n": len(all_ov),
            "h1_ov_mean": np.mean(all_ov) if len(all_ov) > 0 else np.nan,
            "h1_p_halfnorm": chi2_gof(all_ov, "halfnorm") if len(all_ov) >= 20 else np.nan,
            # H2 shape descriptors
            "h2_n": len(pos_td),
            "h2_mean": np.mean(pos_td) if len(pos_td) > 0 else np.nan,
            "h2_median": np.median(pos_td) if len(pos_td) > 0 else np.nan,
            "h2_cv": (np.std(pos_td) / np.mean(pos_td)) if len(pos_td) > 1 and np.mean(pos_td) > 0 else np.nan,
            "h2_skewness": stats.skew(pos_td) if len(pos_td) > 10 else np.nan,
            "h2_gamma_shape": gamma_shape_td,
            "h2_p_gamma": chi2_gof(pos_td, "gamma") if len(pos_td) >= 20 else np.nan,
            "h2_p_beta": chi2_gof(pos_td, "beta") if len(pos_td) >= 20 else np.nan,
            # H3
            "h3_disp_index": disp_idx,
            "h3_disp_p": disp_p,
            # H4
            "h4_n": len(pos_ht),
            "h4_mean": np.mean(pos_ht) if len(pos_ht) > 0 else np.nan,
            "h4_gamma_shape": gamma_shape_ht,
            "h4_p_expon": chi2_gof(pos_ht, "expon") if len(pos_ht) >= 20 else np.nan,
            "h4_p_beta": chi2_gof(pos_ht, "beta") if len(pos_ht) >= 20 else np.nan,
            "h4_p_gamma": chi2_gof(pos_ht, "gamma") if len(pos_ht) >= 20 else np.nan,
        }

        key = (D, dt_val, noise)
        all_overshoots[key] = all_ov
        all_time_dists[key] = pos_td
        records.append(rec)

        done += 1
        if done % 10 == 0 or done == total:
            print("  [%d/%d]" % (done, total))

    return pd.DataFrame(records), all_overshoots, all_time_dists


# ======================================================================
# Plots
# ======================================================================

def plot_hypothesis_summary(df, out_path="figures/hypothesis_summary.pdf"):
    sub = df[(df["dt"] == 1.0) & (df["noise_type"] == "white")].sort_values("D")
    if sub.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # --- H2: gamma shape ---
    ax = axes[0]
    gamma_k = sub["h2_gamma_shape"].fillna(0)
    ax.plot(sub["D"], gamma_k, "o-", color="#9b59b6", markersize=6, linewidth=2)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.annotate("k = 1: exponential", xy=(sub["D"].iloc[-1], 1.05),
                fontsize=8, color="gray", ha="right")
    ax.fill_between(sub["D"], 0, 1, alpha=0.08, color="#9b59b6",
                    label="k < 1: skewed")
    ax.set_xlabel("D")
    ax.set_ylabel("Gamma shape (k)")
    ax.set_title("H2: Форма распределения расстояний\n(k < 1: тяжелый хвост)")
    ax.legend(fontsize=7)

    # --- H3: Poisson DI ---
    ax = axes[1]
    di_vals = sub["h3_disp_index"].values
    colors_h3 = ["#2ecc71" if pd.notna(v) and 0.7 < v < 1.3 else "#e74c3c"
                  for v in di_vals]
    ax.bar(range(len(sub)), [v if pd.notna(v) else 0 for v in di_vals],
           color=colors_h3, edgecolor="white", width=0.7)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5,
               label="Poisson (DI = 1)")
    ax.fill_between(range(len(sub)), 0.7, 1.3, alpha=0.1, color="green")
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(["%.1f" % d for d in sub["D"]], rotation=45, fontsize=8)
    ax.set_xlabel("D")
    ax.set_ylabel("DI (Var/Mean)")
    ax.set_title("H3: Число CP ~ Poisson?\n(зеленый: DI ~ 1)")
    ax.legend(fontsize=7)

    fig.suptitle("Статистический анализ SDE-процесса (dt = 1.0, белый шум)",
                 fontsize=13, fontweight="bold")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("Saved: %s" % out_path)


def plot_overshoot_grid(all_overshoots, D_values, dt_val=1.0,
                        noise_type="white",
                        out_path="figures/overshoot_hist_examples.pdf"):
    D_sel = [0.2, 0.5, 0.75, 1.0, 1.25, 1.5]
    D_sel = [d for d in D_sel if d in D_values]
    n_cols = len(D_sel)
    if n_cols == 0:
        return

    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 4),
                             constrained_layout=True)
    if n_cols == 1:
        axes = [axes]

    for j, D in enumerate(D_sel):
        ax = axes[j]
        key = (D, dt_val, noise_type)
        ov = all_overshoots.get(key, np.array([]))

        if len(ov) < 5:
            ax.text(0.5, 0.5, "n=%d" % len(ov),
                    ha="center", va="center", transform=ax.transAxes)
        else:
            d = ov.astype(float)
            mx = np.percentile(d, 98)
            bins = np.linspace(0, mx, 20)
            ax.hist(d[d <= mx], bins=bins, density=True, alpha=0.65,
                    color="#1f77b4", edgecolor="white", linewidth=0.5)

            x_fit = np.linspace(0.001, mx, 200)
            sigma = np.sqrt(np.mean(d**2))
            ax.plot(x_fit, stats.halfnorm.pdf(x_fit, scale=sigma),
                    "r-", linewidth=2, label="half-normal")

            p_hn = chi2_gof(d, "halfnorm")
            status = "p=%.3f" % p_hn if pd.notna(p_hn) else "N/A"
            if pd.notna(p_hn) and p_hn > 0.05:
                status += " OK"
                ax.patch.set_facecolor("#e8f8e8")

            ax.set_title("D=%.2f, n=%d\n%s" % (D, len(ov), status), fontsize=9)

        ax.set_xlabel("|x - kpi|", fontsize=8)
        if j == 0:
            ax.set_ylabel("Плотность", fontsize=8)
            ax.legend(fontsize=7)
        ax.tick_params(labelsize=6)

    fig.suptitle("H1: Распределение перелётов через границу kpi\n"
                 "dt=1.0, white noise | half-normal фит красным",
                 fontsize=12)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("Saved: %s" % out_path)


def plot_time_dist_grid(all_time_dists, D_values, dt_val=1.0,
                        noise_type="white",
                        out_path="figures/transition_hist_examples.pdf"):
    D_sel = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    D_sel = [d for d in D_sel if d in D_values]
    n_cols = len(D_sel)
    if n_cols == 0:
        return

    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 4),
                             constrained_layout=True)
    if n_cols == 1:
        axes = [axes]

    for j, D in enumerate(D_sel):
        ax = axes[j]
        key = (D, dt_val, noise_type)
        td = all_time_dists.get(key, np.array([]))

        if len(td) < 5:
            ax.text(0.5, 0.5, "n=%d" % len(td),
                    ha="center", va="center", transform=ax.transAxes)
        else:
            mx = np.percentile(td, 95)
            bins = np.linspace(0, mx, 25)
            ax.hist(td[td <= mx], bins=bins, density=True, alpha=0.65,
                    color="#2ca02c", edgecolor="white", linewidth=0.5)

            x_fit = np.linspace(0.5, mx, 200)
            gamma_k = None
            if len(td) >= 20:
                try:
                    a_g, _, sc_g = stats.gamma.fit(td, floc=0)
                    gamma_k = a_g
                    ax.plot(x_fit, stats.gamma.pdf(x_fit, a_g, 0, sc_g),
                            "m-", linewidth=2,
                            label="gamma(k=%.2f)" % a_g)
                except Exception:
                    pass

            title = "D=%.2f, n=%d" % (D, len(td))
            if gamma_k is not None:
                title += ", k=%.2f" % gamma_k
            ax.set_title(title, fontsize=9)

        ax.set_xlabel("Шаги", fontsize=8)
        if j == 0:
            ax.set_ylabel("Плотность", fontsize=8)
            ax.legend(fontsize=7)
        ax.tick_params(labelsize=6)

    fig.suptitle("H2: Расстояния между точками разладки (временные)\n"
                 "dt=1.0, white noise | gamma-фит фиолетовым",
                 fontsize=12)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("Saved: %s" % out_path)


def plot_heatmap(df, col, title, cbar_label, out_path,
                 noise_type="white", cmap="RdYlGn", vmin=0, vmax=1):
    sub = df[df["noise_type"] == noise_type].copy()
    if sub.empty:
        return

    pivot = sub.pivot_table(index="D", columns="dt",
                            values=col, aggfunc="first")
    pivot = pivot.sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    plot_data = pivot.fillna(-0.01)
    im = ax.imshow(plot_data.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(["%.1f" % v for v in pivot.columns], fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(["%.2f" % v for v in pivot.index], fontsize=8)
    ax.set_xlabel("dt", fontsize=11)
    ax.set_ylabel("D", fontsize=11)

    for ii in range(len(pivot.index)):
        for jj in range(len(pivot.columns)):
            val = pivot.iloc[ii, jj]
            if pd.isna(val):
                ax.text(jj, ii, "-", ha="center", va="center",
                        fontsize=7, color="gray")
            else:
                brightness = (val - vmin) / max(vmax - vmin, 1e-9)
                color = "white" if brightness < 0.4 else "black"
                fmt = "%.2f" % val if vmax <= 1 else "%.0f" % val
                ax.text(jj, ii, fmt, ha="center", va="center",
                        fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(cbar_label, fontsize=10)
    ax.set_title(title, fontsize=11)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("Saved: %s" % out_path)


def plot_poisson_heatmap(df, noise_type="white",
                        out_path="figures/poisson_dispersion_heatmap.pdf"):
    from matplotlib.colors import TwoSlopeNorm
    sub = df[df["noise_type"] == noise_type].copy()
    if sub.empty:
        return

    pivot = sub.pivot_table(index="D", columns="dt",
                            values="h3_disp_index", aggfunc="first")
    pivot = pivot.sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    plot_data = pivot.fillna(0)

    from matplotlib.colors import LinearSegmentedColormap
    norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=3)
    im = ax.imshow(plot_data.values, aspect="auto",
                   cmap="coolwarm_r", norm=norm)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(["%.1f" % v for v in pivot.columns], fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(["%.2f" % v for v in pivot.index], fontsize=8)
    ax.set_xlabel("dt", fontsize=11)
    ax.set_ylabel("D", fontsize=11)

    for ii in range(len(pivot.index)):
        for jj in range(len(pivot.columns)):
            val = pivot.iloc[ii, jj]
            if pd.isna(val) or val == 0:
                ax.text(jj, ii, "-", ha="center", va="center",
                        fontsize=7, color="gray")
            else:
                color = "white" if abs(val - 1) > 0.8 else "black"
                ax.text(jj, ii, "%.1f" % val, ha="center", va="center",
                        fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("DI (Var/Mean)", fontsize=10)
    ax.set_title("H3: Индекс дисперсии числа CP\n"
                 "Белое (DI = 1) = Пуассон | Цветное = отклонение",
                 fontsize=11)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("Saved: %s" % out_path)


# ======================================================================
# Main
# ======================================================================

def main():
    D_values = np.unique(np.round(np.concatenate([
        np.arange(0.1, 0.55, 0.1),
        np.arange(0.5, 2.05, 0.1),
    ]), 2))

    dt_values = np.round(np.linspace(0.2, 3.0, 15), 2)
    noise_types = ["white"]

    total = len(D_values) * len(dt_values)
    print("Grid: %d D x %d dt = %d combos" % (len(D_values), len(dt_values), total))
    print("D:  %s" % D_values)
    print("dt: %s" % dt_values)

    print("\nRunning...")
    df, all_ov, all_td = run_grid(
        D_values, dt_values, noise_types,
        length=10000, n_series=30, base_seed=0
    )

    out_dir = "results/transition_analysis"
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    print("\nSaved: %s/summary.csv" % out_dir)

    # Report
    print("\n" + "=" * 60)
    sub1 = df[(df["dt"] == 1.0) & (df["noise_type"] == "white")].sort_values("D")
    print("dt=1.0 white noise summary:")
    print("%-6s %6s %8s %8s %8s %8s" % ("D", "CP/ser", "ov_p_hn", "gam_k", "disp_idx", "ht_gam_k"))
    for _, r in sub1.iterrows():
        print("%-6.2f %6.0f %8s %8s %8s %8s" % (
            r["D"], r["mean_cp"],
            "%.3f" % r["h1_p_halfnorm"] if pd.notna(r["h1_p_halfnorm"]) else "N/A",
            "%.2f" % r["h2_gamma_shape"] if pd.notna(r["h2_gamma_shape"]) else "N/A",
            "%.2f" % r["h3_disp_index"] if pd.notna(r["h3_disp_index"]) else "N/A",
            "%.2f" % r["h4_gamma_shape"] if pd.notna(r["h4_gamma_shape"]) else "N/A",
        ))

    h1_pass = df[df["h1_p_halfnorm"] > 0.05]
    print("\nH1 (ov ~ halfnorm): %d/%d pass, D range: [%.2f, %.2f]" % (
        len(h1_pass), len(df),
        h1_pass["D"].min() if not h1_pass.empty else 0,
        h1_pass["D"].max() if not h1_pass.empty else 0))

    for col_name, label in [("h2_p_gamma", "gamma"), ("h2_p_beta", "beta"),
                             ("h4_p_expon", "expon"), ("h4_p_beta", "beta(ht)")]:
        n_pass = len(df[df[col_name] > 0.05]) if col_name in df.columns else 0
        print("  %s: %d/%d pass" % (label, n_pass, len(df)))

    print("\nGenerating plots...")
    plot_hypothesis_summary(df)
    plot_overshoot_grid(all_ov, D_values, dt_val=1.0)
    plot_time_dist_grid(all_td, D_values, dt_val=1.0)

    plot_heatmap(df, "h1_p_halfnorm",
                 "H1: Перелёт ~ half-normal (p-value)\nЗелёное = фит (p > 0.05)",
                 "p-value", "figures/overshoot_halfnorm_heatmap.pdf")

    plot_heatmap(df, "mean_cp",
                 "Среднее число CP на ряд (length=10000)",
                 "CP / ряд", "figures/mean_cp_heatmap.pdf",
                 cmap="YlOrRd", vmin=0,
                 vmax=max(df["mean_cp"].quantile(0.9), 1))

    plot_poisson_heatmap(df)

    print("\nDone!")


if __name__ == "__main__":
    main()
