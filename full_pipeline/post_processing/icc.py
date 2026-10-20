"""Reproducibility of radiomic features with the intraclass correlation coefficient (ICC).

Reference: Shrout PE, Fleiss JL. Intraclass correlations: uses in assessing rater reliability.
           Psychological Bulletin 1979; 86(2): 420-428.

Data: for every feature an n x k table: n subjects (breast ROIs) x k "raters". Here the raters are the
conditions under which the feature was extracted: the baseline (the input feature CSV) and the perturbed
extractions of reproducibility_analysis.ipynb (outputs/reproducibility/<perturbation>/<task>.json).

Two-way ANOVA mean squares (Shrout & Fleiss, Table 2):
    BMS (MSR)  between subjects       WMS (MSW)  within subjects
    JMS (MSC)  between raters         EMS (MSE)  residual
Six ICC forms (Shrout & Fleiss, Table 3):
    ICC(1,1) = (BMS - WMS) / (BMS + (k-1) WMS)                        one-way random
    ICC(2,1) = (BMS - EMS) / (BMS + (k-1) EMS + k (JMS - EMS) / n)    two-way random, absolute agreement
    ICC(3,1) = (BMS - EMS) / (BMS + (k-1) EMS)                        two-way mixed, consistency
    ICC(1,k) = (BMS - WMS) / BMS
    ICC(2,k) = (BMS - EMS) / (BMS + (JMS - EMS) / n)
    ICC(3,k) = (BMS - EMS) / BMS
ICC(2,1) is used to judge reproducibility (Koo & Li 2016: < 0.5 poor, 0.5-0.75 moderate, 0.75-0.9 good,
> 0.9 excellent). ICC is computed separately for every MRI channel (subjects = breasts of that channel).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f as f_distribution

ID_COLUMNS = ["case", "side", "channel", "task"]
PERTURBATIONS = ["volume_minus10", "volume_plus10", "n4", "noise"]
CLASS_ORDER = ["shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]
CATEGORIES = [-np.inf, 0.5, 0.75, 0.9, np.inf]
CATEGORY_NAMES = ["poor (< 0.5)", "moderate (0.5-0.75)", "good (0.75-0.9)", "excellent (> 0.9)"]
CATEGORY_COLOURS = ["#d73027", "#fc8d59", "#91bfdb", "#4575b4"]


#ICC formulas
def mean_squares(data):
    """Two-way ANOVA mean squares of data with shape (n subjects, k raters, features)."""
    n, k = data.shape[0], data.shape[1]
    grand = data.mean(axis=(0, 1))
    ss_total = ((data - grand) ** 2).sum(axis=(0, 1))
    ss_subjects = k * ((data.mean(axis=1) - grand) ** 2).sum(axis=0)
    ss_raters = n * ((data.mean(axis=0) - grand) ** 2).sum(axis=0)
    ss_error = ss_total - ss_subjects - ss_raters
    return {"BMS": ss_subjects / (n - 1), "WMS": (ss_total - ss_subjects) / (n * (k - 1)),
            "JMS": ss_raters / (k - 1), "EMS": ss_error / ((n - 1) * (k - 1)), "n": n, "k": k}


def icc_values(ms):
    """The six ICC forms of Shrout & Fleiss (1979)."""
    BMS, WMS, JMS, EMS, n, k = ms["BMS"], ms["WMS"], ms["JMS"], ms["EMS"], ms["n"], ms["k"]
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"ICC(1,1)": (BMS - WMS) / (BMS + (k - 1) * WMS),
                "ICC(2,1)": (BMS - EMS) / (BMS + (k - 1) * EMS + k * (JMS - EMS) / n),
                "ICC(3,1)": (BMS - EMS) / (BMS + (k - 1) * EMS),
                "ICC(1,k)": (BMS - WMS) / BMS,
                "ICC(2,k)": (BMS - EMS) / (BMS + (JMS - EMS) / n),
                "ICC(3,k)": (BMS - EMS) / BMS}


def icc21_confidence_interval(ms, icc21, alpha=0.05):
    """95 % confidence interval of ICC(2,1) (Shrout & Fleiss 1979, with Satterthwaite's degrees of freedom)."""
    BMS, JMS, EMS, n, k = ms["BMS"], ms["JMS"], ms["EMS"], ms["n"], ms["k"]
    with np.errstate(divide="ignore", invalid="ignore"):
        Fj = JMS / EMS
        vn = (k - 1) * (n - 1) * (k * icc21 * Fj + n * (1 + (k - 1) * icc21) - k * icc21) ** 2
        vd = (n - 1) * k ** 2 * icc21 ** 2 * Fj ** 2 + (n * (1 + (k - 1) * icc21) - k * icc21) ** 2
        v = vn / vd
        F_low = f_distribution.ppf(1 - alpha / 2, n - 1, v)
        F_high = f_distribution.ppf(1 - alpha / 2, v, n - 1)
        lower = n * (BMS - F_low * EMS) / (F_low * (k * JMS + (k * n - k - n) * EMS) + n * BMS)
        upper = n * (F_high * BMS - EMS) / (k * JMS + (k * n - k - n) * EMS + n * F_high * BMS)
    return lower, upper


def shrout_fleiss_example():
    """Table 4 of Shrout & Fleiss: 6 targets x 4 judges. Published ICCs: .17 .29 .71 .44 .62 .91."""
    data = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]], float)
    ms = mean_squares(data)
    values = icc_values(ms)
    low, high = icc21_confidence_interval(ms, values["ICC(2,1)"])
    table = pd.DataFrame({"ours": list(values.values()), "Shrout & Fleiss": [0.17, 0.29, 0.71, 0.44, 0.62, 0.91]},
                         index=list(values)).round(2)
    print("ICC(2,1) 95%% CI: %.2f to %.2f" % (low, high))
    return table


#data
def load_conditions(csv_path, repro_dir, perturbations=PERTURBATIONS):
    """{condition: DataFrame} with the same rows (tasks) and columns (features) as the CSV.

    'baseline' is the CSV itself; every perturbation is read from <repro_dir>/<perturbation>/<task>.json.
    Features that are not in the perturbed files (e.g. Gabor, Laws) are returned separately.
    """
    baseline = pd.read_csv(csv_path)
    features = [c for c in baseline.columns if c not in ID_COLUMNS]
    first = json.loads((Path(repro_dir) / perturbations[0] / f"{baseline['task'].iloc[0]}.json").read_text())
    available = [c for c in features if c in first["features"]]
    missing = [c for c in features if c not in first["features"]]

    conditions = {"baseline": baseline[ID_COLUMNS + available]}
    for perturbation in perturbations:
        rows = []
        for task in baseline["task"]:
            payload = json.loads((Path(repro_dir) / perturbation / f"{task}.json").read_text())
            rows.append([payload["features"][c] for c in available])
        conditions[perturbation] = pd.concat([baseline[ID_COLUMNS], pd.DataFrame(rows, columns=available)], axis=1)
    return conditions, missing


def describe_features(columns):
    """Image type, class and name of every feature column '<image type>_<class>_<feature>'."""
    parts = [c.split("_", 2) for c in columns]
    return pd.DataFrame({"feature": columns, "image_type": [p[0] for p in parts],
                         "feature_class": [p[1] for p in parts], "feature_name": [p[2] for p in parts]})


#ICC tables
def icc_per_channel(conditions, perturbations=PERTURBATIONS):
    """One row per (feature, channel): all six ICCs, the 95 % CI of ICC(2,1) and ICC(2,1) baseline vs each perturbation."""
    names = list(conditions)
    features = [c for c in conditions["baseline"].columns if c not in ID_COLUMNS]
    tables = []
    for channel in sorted(conditions["baseline"]["channel"].unique()):
        rows = conditions["baseline"]["channel"] == channel
        # data shape: (subjects, conditions, features)
        data = np.stack([conditions[name].loc[rows, features].to_numpy(float) for name in names], axis=1)
        ms = mean_squares(data)
        table = describe_features(features)
        table.insert(1, "channel", channel)
        for form, values in icc_values(ms).items():
            table[form] = values
        table["ICC(2,1) CI low"], table["ICC(2,1) CI high"] = icc21_confidence_interval(ms, table["ICC(2,1)"].to_numpy())
        for perturbation in perturbations:
            pair = data[:, [0, names.index(perturbation)], :]
            table[f"ICC(2,1) vs {perturbation}"] = icc_values(mean_squares(pair))["ICC(2,1)"]
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def icc_per_feature(per_channel, threshold=0.9):
    """One row per feature: lowest and median ICC(2,1) over the channels, category and reproducible flag."""
    summary = per_channel.groupby(["feature", "image_type", "feature_class", "feature_name"], sort=False).agg(
        icc21_min=("ICC(2,1)", "min"), icc21_median=("ICC(2,1)", "median"),
        icc21_ci_low_min=("ICC(2,1) CI low", "min")).reset_index()
    summary["category"] = pd.cut(summary["icc21_min"], CATEGORIES, labels=CATEGORY_NAMES)
    summary["reproducible"] = summary["icc21_min"] >= threshold        # every channel must pass
    return summary.sort_values("icc21_min").reset_index(drop=True)


#figures
def plot_overview(per_feature, title, threshold):
    """Histogram of the per-feature ICC(2,1) and the number of features per ICC category."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    values = per_feature["icc21_min"].dropna().clip(lower=-0.2)
    axes[0].hist(values, bins=50, color="#4575b4")
    for edge, colour in zip([0.5, 0.75, 0.9], CATEGORY_COLOURS[1:]):
        axes[0].axvline(edge, color=colour, ls=":", lw=1.5)
    axes[0].axvline(threshold, color="k", ls="--", lw=1.2, label=f"threshold {threshold}")
    axes[0].set_xlabel("ICC(2,1), lowest over the MRI channels")
    axes[0].set_ylabel("number of features")
    axes[0].set_title("Distribution of feature reproducibility")
    axes[0].legend()

    counts = per_feature["category"].value_counts().reindex(CATEGORY_NAMES).fillna(0)
    axes[1].barh(CATEGORY_NAMES, counts, color=CATEGORY_COLOURS)
    for i, count in enumerate(counts):
        axes[1].text(count, i, f" {int(count)} ({100 * count / max(len(per_feature), 1):.0f}%)", va="center")
    axes[1].set_xlabel("number of features")
    axes[1].set_title("ICC categories")
    axes[1].set_xlim(0, counts.max() * 1.3 + 1)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_class_and_image_type(per_feature, title, threshold):
    """Heat maps over image type x feature class: median ICC(2,1) and % of reproducible features."""
    types = sorted(per_feature["image_type"].unique())
    classes = [c for c in CLASS_ORDER if c in set(per_feature["feature_class"])]
    median = per_feature.pivot_table(index="image_type", columns="feature_class", values="icc21_min",
                                     aggfunc="median").reindex(index=types, columns=classes)
    percent = (per_feature.pivot_table(index="image_type", columns="feature_class", values="reproducible",
                                       aggfunc="mean") * 100).reindex(index=types, columns=classes)
    height = max(3, 0.5 * len(types) + 2)
    fig, axes = plt.subplots(1, 2, figsize=(15, height))
    sns.heatmap(median.astype(float), ax=axes[0], cmap="RdYlBu", vmin=0, vmax=1, annot=True, fmt=".2f",
                cbar_kws={"label": "median ICC(2,1)"})
    axes[0].set_title("Median ICC(2,1) per image type and feature class")
    sns.heatmap(percent.astype(float), ax=axes[1], cmap="viridis", vmin=0, vmax=100, annot=True, fmt=".0f",
                cbar_kws={"label": "% of features"})
    axes[1].set_title(f"% of features with ICC(2,1) >= {threshold} in every channel")
    for ax in axes:
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_perturbations_and_channels(per_channel, title, threshold, perturbations=PERTURBATIONS):
    """Box plots: ICC(2,1) of the baseline vs each single perturbation, and ICC(2,1) per MRI channel and class."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    pairwise = per_channel[[f"ICC(2,1) vs {p}" for p in perturbations]].clip(lower=-0.2)
    pairwise.columns = perturbations
    sns.boxplot(data=pairwise, ax=axes[0], color="#91bfdb", fliersize=2)
    axes[0].axhline(threshold, color="k", ls="--", lw=1)
    axes[0].set_ylabel("ICC(2,1) baseline vs perturbation")
    axes[0].set_title("Which perturbation changes the features most?")

    data = per_channel.assign(ICC=per_channel["ICC(2,1)"].clip(lower=-0.2))
    order = [c for c in CLASS_ORDER if c in set(data["feature_class"])]
    sns.boxplot(data=data, x="feature_class", y="ICC", hue="channel", order=order, ax=axes[1], fliersize=2)
    axes[1].axhline(threshold, color="k", ls="--", lw=1)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("ICC(2,1), all conditions")
    axes[1].set_title("Reproducibility per feature class and MRI channel")
    axes[1].legend(title="channel", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_least_reproducible(per_channel, title, threshold, n=25):
    """The n features with the lowest ICC(2,1) (worst channel), with the 95 % confidence interval."""
    defined = per_channel.dropna(subset=["ICC(2,1)"])          # constant features have no ICC
    worst = defined.loc[defined.groupby("feature")["ICC(2,1)"].idxmin()]
    worst = worst.sort_values("ICC(2,1)").head(n).iloc[::-1]
    labels = [f"{f}  (ch{c})" for f, c in zip(worst["feature"], worst["channel"])]
    values = worst["ICC(2,1)"].clip(lower=-0.2).to_numpy()
    low = worst["ICC(2,1) CI low"].clip(lower=-0.2).to_numpy()
    high = worst["ICC(2,1) CI high"].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 0.32 * len(worst) + 1.5))
    colours = [CATEGORY_COLOURS[np.searchsorted([0.5, 0.75, 0.9], v, side="right")] for v in values]
    ax.barh(range(len(worst)), values, color=colours)
    ax.errorbar(values, range(len(worst)), xerr=[values - low, np.maximum(high - values, 0)], fmt="none",
                ecolor="k", lw=0.8, capsize=2)
    ax.axvline(threshold, color="k", ls="--", lw=1)
    ax.set_yticks(range(len(worst)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("ICC(2,1) with 95 % CI (worst channel)")
    ax.set_title(f"{title}: {len(worst)} least reproducible features")
    fig.tight_layout()
    return fig


# whole analysis
def run_icc_analysis(csv_path, repro_dir, output_dir, threshold=0.9, perturbations=PERTURBATIONS):
    """Compute the ICC of every feature of one CSV, save tables and figures to <output_dir>/<csv name>/."""
    csv_path = Path(csv_path)
    out = Path(output_dir) / csv_path.stem
    out.mkdir(parents=True, exist_ok=True)

    conditions, missing = load_conditions(csv_path, repro_dir, perturbations)
    results = {"missing_features": missing, "figures": {}}
    if len(conditions["baseline"].columns) == len(ID_COLUMNS):
        print(f"{csv_path.name}: none of the {len(missing)} features is in the perturbed extractions - nothing to do")
        return results

    per_channel = icc_per_channel(conditions, perturbations)
    per_feature = icc_per_feature(per_channel, threshold)
    results.update(per_channel=per_channel, per_feature=per_feature)

    name = csv_path.stem
    results["figures"]["overview"] = plot_overview(per_feature, name, threshold)
    results["figures"]["class_and_image_type"] = plot_class_and_image_type(per_feature, name, threshold)
    results["figures"]["perturbations_and_channels"] = plot_perturbations_and_channels(per_channel, name, threshold,
                                                                                       perturbations)
    results["figures"]["least_reproducible"] = plot_least_reproducible(per_channel, name, threshold)

    per_channel.to_csv(out / "icc_per_channel.csv", index=False)
    per_feature.to_csv(out / "icc_per_feature.csv", index=False)
    per_feature.loc[per_feature["reproducible"], "feature"].to_csv(out / "reproducible_features.csv", index=False)
    for fig_name, fig in results["figures"].items():
        fig.savefig(out / f"{fig_name}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)                      # shown later with display(); avoids showing every figure twice

    print(f"{name}: {len(per_feature)} features ({len(missing)} without perturbed data) | "
          f"{int(per_feature['reproducible'].sum())} reproducible (ICC(2,1) >= {threshold} in every channel) | "
          f"saved to {out}")
    return results
