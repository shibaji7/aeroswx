"""Reusable publication figures (built against tidy-schema frames; see ``sample_data``).

Design notes: one y-scale per axes (no dual-axis by default -- ``plot_timeseries`` stacks the
delay and the index on a shared time axis; ``layout="twin"`` gives the overlaid variant),
thin marks, recessive grid, text in ink colour, 95 % confidence error bars.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from swe_airline_delay.plotting import style as st

CI_Z = 1.96

# (dlon, dlat, ha) label offsets (deg) for airports that would otherwise overlap (NYC area)
_LABEL_OFFSETS = {
    "EWR": (-0.7, -0.9, "right"),
    "JFK": (0.7, -1.1, "left"),
    "LGA": (0.7, 0.4, "left"),
}


def _sem_ci(x: pd.Series) -> tuple[float, float, int]:
    x = x.dropna()
    n = len(x)
    if n == 0:
        return np.nan, np.nan, 0
    sem = x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return float(x.mean()), float(CI_Z * sem) if n > 1 else np.nan, n


# --------------------------------------------------------------------------- time series
def plot_timeseries(
    delay_df: pd.DataFrame,
    index_df: pd.DataFrame,
    index_name: str,
    time_col: str = "scheduled_dep",
    delay_col: str = "dep_delay_min",
    freq: str = "1h",
    layout: str = "stacked",
    index_label: str | None = None,
):
    """Mean delay (minutes) and a space-weather index vs. UTC time.

    ``layout="stacked"`` (default): two panels sharing the time axis. ``layout="twin"``: both on one
    axes with a secondary y-axis (overlay). Returns ``(fig, axes)``.
    """
    with st.style_context():
        d = delay_df.dropna(subset=[delay_col]).set_index(
            pd.to_datetime(delay_df.loc[delay_df[delay_col].notna(), time_col], utc=True)
        )
        delay = d[delay_col].resample(freq).mean()
        idx = index_df[index_name]
        ilabel = index_label or index_name
        if layout == "twin":
            fig, ax = plt.subplots()
            ax.plot(delay.index, delay.values, color=st.DEEP_BLUE, label="Mean departure delay")
            ax2 = ax.twinx()
            ax2.spines["right"].set_visible(True)
            ax2.grid(False)
            ax2.plot(
                idx.index,
                idx.values,
                color=st.GOLDEN_YELLOW,
                path_effects=st.yellow_outline(),
                label=ilabel,
            )
            ax.set_ylabel("Mean departure delay (min)")
            ax2.set_ylabel(ilabel)
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="upper left")
            axes = (ax, ax2)
        else:
            fig, axes = plt.subplots(
                2, 1, sharex=True, figsize=(6.8, 5.0), gridspec_kw={"height_ratios": [1.2, 1]}
            )
            axes[0].plot(
                delay.index, delay.values, color=st.DEEP_BLUE, label="Mean departure delay"
            )
            axes[0].set_ylabel("Mean departure delay (min)")
            axes[1].fill_between(
                idx.index, idx.values, color=st.GOLDEN_YELLOW, alpha=0.55, linewidth=0
            )
            axes[1].plot(idx.index, idx.values, color=st.DARK_GOLD, lw=1.2, label=ilabel)
            axes[1].set_ylabel(ilabel)
            axes[1].set_xlabel("Time (UTC)")
            axes[0].legend(loc="upper left")
            axes[1].legend(loc="upper left")
        fig.autofmt_xdate()
        fig.tight_layout()
    return fig, axes


# --------------------------------------------------------------------------- dose response
def plot_dose_response(
    delay_df: pd.DataFrame,
    index_name: str,
    bins: int | list[float] = 8,
    metric: str = "dep_delay_min",
    ax=None,
):
    """Binned mean ``metric`` vs. ``index_name`` with 95 % CI error bars.

    ``delay_df`` must already contain ``index_name`` (e.g. the aligned panel). ``metric`` is a
    delay column (mean minutes) or ``"cancelled"`` (rate, normal-approximation CI). ``bins`` is a
    number of equal-count (quantile) bins or explicit edges. Returns ``(fig, ax)``.
    """
    df = delay_df.dropna(subset=[index_name]).copy()
    if metric == "cancelled":
        df[metric] = df[metric].astype(float)
    cut = (
        pd.qcut(df[index_name], bins, duplicates="drop")
        if isinstance(bins, int)
        else pd.cut(df[index_name], bins)
    )
    g = df.groupby(cut, observed=True)
    stats = g[metric].apply(lambda s: pd.Series(_sem_ci(s), index=["mean", "ci", "n"])).unstack()
    x = g[index_name].mean()
    with st.style_context():
        fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
        ax.errorbar(
            x,
            stats["mean"],
            yerr=stats["ci"],
            fmt="o-",
            color=st.DEEP_BLUE,
            mfc=st.GOLDEN_YELLOW,
            mec=st.DEEP_BLUE,
            ecolor=st.DEEP_BLUE,
            capsize=3,
            label="Bin mean $\\pm$ 95% CI",
        )
        ax.set_xlabel(index_name)
        ax.set_ylabel(
            "Cancellation rate" if metric == "cancelled" else "Mean departure delay (min)"
        )
        ax.legend(loc="upper left")
        fig.tight_layout()
    return fig, ax


# --------------------------------------------------------------------------- superposed epoch
def plot_superposed_epoch(
    event_times,
    delay_df: pd.DataFrame,
    window: tuple[int, int] = (-3, 6),
    time_col: str = "scheduled_dep",
    delay_col: str = "dep_delay_min",
    ax=None,
):
    """Superposed-epoch curve of hourly mean delay around ``event_times`` (epoch hour 0 = event).

    For each event the flights are binned by whole hours relative to the event; the plotted line is
    the mean across events with a 95 % CI band (events are the replicate unit).
    Returns ``(fig, ax)``.
    """
    lo, hi = window
    t = pd.to_datetime(delay_df[time_col], utc=True)
    ok = delay_df[delay_col].notna()
    t, y = t[ok].reset_index(drop=True), delay_df.loc[ok, delay_col].reset_index(drop=True)
    epochs = np.arange(lo, hi + 1)
    curves = []
    for ev in pd.to_datetime(pd.Index(event_times), utc=True):
        h = np.floor((t - ev).dt.total_seconds() / 3600.0)
        sel = (h >= lo) & (h <= hi)
        curves.append(y[sel].groupby(h[sel].astype(int)).mean().reindex(epochs))
    mat = pd.DataFrame(curves)
    mean = mat.mean()
    n = mat.count()
    ci = CI_Z * mat.std(ddof=1) / np.sqrt(n.where(n > 1))
    with st.style_context():
        fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
        ax.axvline(0, color=st.INK, lw=0.8, ls="--")
        ax.fill_between(
            epochs,
            mean - ci,
            mean + ci,
            color=st.GOLDEN_YELLOW,
            alpha=0.5,
            linewidth=0,
            label="95% CI",
        )
        ax.plot(
            epochs,
            mean,
            "o-",
            color=st.DEEP_BLUE,
            mfc=st.GOLDEN_YELLOW,
            label=f"Mean of {len(mat)} events",
        )
        ax.set_xlabel("Hours relative to event onset")
        ax.set_ylabel("Mean departure delay (min)")
        ax.set_xticks(epochs)
        ax.legend(loc="upper left")
        fig.tight_layout()
    return fig, ax


# --------------------------------------------------------------------------- map
def plot_airport_map(
    airports_df: pd.DataFrame,
    color_by: str | None = None,
    draw_features: bool = True,
    label: bool = True,
    cmap=None,
):
    """US map (cartopy) of the airports; optional colouring by a column (e.g. delay burden).

    Airports north of 50 N (ANC) go in an Alaska inset. ``draw_features=False`` skips coastlines and
    state borders (they are downloaded from Natural Earth on first draw -- free, but needs network).
    Returns ``(fig, (ax_main, ax_inset))``.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    cmap = cmap or st.BLUE_CMAP
    proj = ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=37.5)
    pc = ccrs.PlateCarree()
    df = airports_df.copy()
    vals = df[color_by] if color_by else None
    norm = plt.Normalize(vals.min(), vals.max()) if vals is not None else None

    with st.style_context():
        fig = plt.figure(figsize=(8.0, 5.2))
        ax = fig.add_axes([0.02, 0.05, 0.86 if color_by else 0.96, 0.9], projection=proj)
        ax.set_extent([-125, -66.5, 24, 50], crs=pc)
        inset = fig.add_axes(
            [0.03, 0.06, 0.22, 0.26],
            projection=ccrs.AlbersEqualArea(central_longitude=-150, central_latitude=62),
        )
        inset.set_extent([-170, -130, 52, 72], crs=pc)
        for a in (ax, inset):
            a.set_facecolor(st.WHITE)
            a.spines["geo"].set_edgecolor("#4A4A4A")
            a.spines["geo"].set_linewidth(0.6)
            if draw_features:
                a.add_feature(
                    cfeature.LAND.with_scale("50m"), facecolor=st.LIGHT_GREY, edgecolor="none"
                )
                a.add_feature(
                    cfeature.COASTLINE.with_scale("50m"), edgecolor="#9AA5B1", linewidth=0.5
                )
                a.add_feature(
                    cfeature.STATES.with_scale("50m"),
                    edgecolor="#C9D1D9",
                    linewidth=0.4,
                    facecolor="none",
                )
        sc = None
        for a, sel in ((ax, df["lat"] <= 50), (inset, df["lat"] > 50)):
            sub = df[sel]
            if sub.empty:
                continue
            if color_by:
                sc = a.scatter(
                    sub["lon"],
                    sub["lat"],
                    c=sub[color_by],
                    cmap=cmap,
                    norm=norm,
                    s=60,
                    edgecolor=st.DEEP_BLUE,
                    linewidth=0.8,
                    transform=pc,
                    zorder=5,
                )
            else:
                sc = a.scatter(
                    sub["lon"],
                    sub["lat"],
                    s=60,
                    facecolor=st.GOLDEN_YELLOW,
                    edgecolor=st.DEEP_BLUE,
                    linewidth=0.9,
                    transform=pc,
                    zorder=5,
                )
            if label:
                for r in sub.itertuples():
                    dx, dy, ha = _LABEL_OFFSETS.get(r.iata, (0.6, 0.5, "left"))
                    a.text(
                        r.lon + dx,
                        r.lat + dy,
                        r.iata,
                        transform=pc,
                        fontsize=7.5,
                        color=st.INK,
                        ha=ha,
                        zorder=6,
                    )
        if not (df["lat"] > 50).any():
            inset.remove()
        if color_by and sc is not None:
            cax = fig.add_axes([0.9, 0.2, 0.015, 0.6])
            fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label=color_by)
    return fig, (ax, inset)


if __name__ == "__main__":  # python -m swe_airline_delay.plotting.figures  -> demo PNGs
    from pathlib import Path

    from swe_airline_delay.config import load_airports
    from swe_airline_delay.plotting.sample_data import make_synthetic

    out = Path("data/processed/figures_demo")
    out.mkdir(parents=True, exist_ok=True)
    d, i, ev = make_synthetic()
    plot_timeseries(d, i, "kp")[0].savefig(out / "timeseries.png")
    p = d.assign(t=d["scheduled_dep"].dt.floor("1h")).merge(
        i[["kp"]], left_on="t", right_index=True
    )
    plot_dose_response(p, "kp")[0].savefig(out / "dose_response.png")
    plot_superposed_epoch(ev, d)[0].savefig(out / "superposed_epoch.png")
    a = load_airports().assign(lat_exposure=lambda x: x["lat"])
    plot_airport_map(a, "lat_exposure")[0].savefig(out / "airport_map.png")
    print("wrote", out)
