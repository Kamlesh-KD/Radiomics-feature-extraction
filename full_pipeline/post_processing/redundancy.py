"""Redundancy and association of radiomic features, measured with the Spearman rank correlation.

Input: a feature CSV with one row per sample and one column per feature, named '<image type>_<class>_<feature>'
(e.g. features_raw_image.csv). Columns case / side / channel / task are treated as identifiers.

    redundancy   two features are redundant if |rho| >= threshold. Features are grouped with complete-linkage
                 clustering on the distance 1 - |rho|, so every pair inside a group has |rho| >= threshold;
                 one representative per group is kept.
    association  mean |rho| between all features of two feature classes (or image types). The diagonal shows how
                 redundant a class is internally, the off-diagonal how strongly two classes are associated.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

ID_COLUMNS = ["case", "side", "channel", "task"]
CLASS_ORDER = ["shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]


def load_feature_matrix(csv_path, channel=None):
    """Feature matrix (rows = samples, columns = features) of a CSV; optionally only one MRI channel.

    Returns (features, dropped): constant columns are dropped because their correlation is undefined.
    """
    table = pd.read_csv(csv_path)
    if channel is not None:
        table = table[table["channel"] == channel]
    features = table.drop(columns=[c for c in ID_COLUMNS if c in table.columns]).astype(float)
    features = features.dropna(axis=1, how="any")
    constant = [c for c in features.columns if features[c].nunique() <= 1]
    return features.drop(columns=constant), constant


def describe_features(columns):
    """Table with image type, class and name of every feature column."""
    parts = [c.split("_", 2) for c in columns]
    return pd.DataFrame({"feature": columns, "image_type": [p[0] for p in parts],
                         "feature_class": [p[1] for p in parts], "feature_name": [p[2] for p in parts]})


def spearman_matrix(features):
    """Spearman rank correlation between every pair of features."""
    return features.corr(method="spearman")


def redundant_pairs(rho, threshold=0.9):
    """All feature pairs with |rho| >= threshold, strongest first."""
    upper = rho.where(np.triu(np.ones(rho.shape, dtype=bool), k=1)).stack()
    pairs = upper[upper.abs() >= threshold].rename("rho").reset_index()
    pairs.columns = ["feature_1", "feature_2", "rho"]
    info = describe_features(rho.columns).set_index("feature")
    pairs["class_1"] = info.loc[pairs["feature_1"], "feature_class"].to_numpy()
    pairs["class_2"] = info.loc[pairs["feature_2"], "feature_class"].to_numpy()
    pairs["same_class"] = pairs["class_1"] == pairs["class_2"]
    return pairs.reindex(pairs["rho"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def redundancy_clusters(rho, threshold=0.9):
    """Group features so that every pair in a group has |rho| >= threshold; mark one representative per group.

    The representative is the feature with the highest mean |rho| to the other members of its group.
    """
    distance = (1 - rho.abs()).clip(lower=0).to_numpy(copy=True)
    np.fill_diagonal(distance, 0)
    labels = fcluster(linkage(squareform(distance, checks=False), method="complete"),
                      t=1 - threshold, criterion="distance")
    clusters = describe_features(rho.columns)
    clusters["cluster"] = labels
    clusters["cluster_size"] = clusters.groupby("cluster")["feature"].transform("size")
    clusters["representative"] = False
    for _, members in clusters.groupby("cluster"):
        names = members["feature"].tolist()
        mean_abs = rho.loc[names, names].abs().mean(axis=1)
        clusters.loc[members.index[names.index(mean_abs.idxmax())], "representative"] = True
    return clusters.sort_values(["cluster", "representative"], ascending=[True, False]).reset_index(drop=True)


def group_association(rho, groups, threshold=0.9, order=None):
    """Mean |rho| and % of pairs with |rho| >= threshold between every two groups (classes or image types).

    groups: one group name per feature column. Pairs of a feature with itself are left out.
    """
    groups = pd.Series(list(groups), index=rho.columns)
    names = [g for g in (order or []) if g in set(groups)] + sorted(set(groups) - set(order or []))
    absolute = rho.abs().to_numpy(copy=True)
    np.fill_diagonal(absolute, np.nan)
    mean_abs = pd.DataFrame(index=names, columns=names, dtype=float)
    percent_redundant = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            block = absolute[np.ix_((groups == a).to_numpy(), (groups == b).to_numpy())]
            values = block[~np.isnan(block)]
            mean_abs.loc[a, b] = values.mean() if values.size else np.nan
            percent_redundant.loc[a, b] = 100 * (values >= threshold).mean() if values.size else np.nan
    return mean_abs, percent_redundant


def class_summary(clusters):
    """Per feature class: number of features, number kept after redundancy removal, % redundant."""
    summary = clusters.groupby("feature_class").agg(features=("feature", "size"), kept=("representative", "sum"))
    summary["removed"] = summary["features"] - summary["kept"]
    summary["percent_redundant"] = 100 * summary["removed"] / summary["features"]
    return summary.reindex([c for c in CLASS_ORDER if c in summary.index])


# --------------------------------------------------------------------------------------------- figures
def class_palette(classes):
    """Fixed colour per feature class."""
    colours = dict(zip(CLASS_ORDER, sns.color_palette("tab10", len(CLASS_ORDER))))
    return {c: colours.get(c, (0.5, 0.5, 0.5)) for c in classes}


def plot_correlation_clustermap(rho, title):
    """Clustered Spearman matrix; the colour bar on the left shows the feature class of every row."""
    info = describe_features(rho.columns)
    palette = class_palette(info["feature_class"].unique())
    row_colours = pd.Series(info["feature_class"].map(palette).to_numpy(), index=rho.columns, name="class")
    show_labels = len(rho) <= 60
    grid = sns.clustermap(rho, method="average", metric="euclidean", cmap="RdBu_r", vmin=-1, vmax=1,
                          row_colors=row_colours, xticklabels=show_labels, yticklabels=show_labels,
                          figsize=(12, 11), dendrogram_ratio=0.12, cbar_pos=(0.02, 0.93, 0.1, 0.015),
                          cbar_kws={"label": "Spearman rho", "orientation": "horizontal"})
    grid.ax_heatmap.set_xlabel("")
    grid.ax_heatmap.set_ylabel("")
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[c]) for c in palette]
    grid.ax_heatmap.legend(handles, list(palette), title="feature class", loc="upper left",
                           bbox_to_anchor=(1.01, 1.0), fontsize=8)
    grid.figure.suptitle(f"{title}: Spearman correlation of {len(rho)} features (clustered)", y=1.01)
    return grid.figure


def plot_group_association(mean_abs, percent_redundant, title, group_label, threshold):
    """Two heat maps: mean |rho| and % redundant pairs between groups."""
    size = max(5, 0.55 * len(mean_abs) + 2.5)
    fig, axes = plt.subplots(1, 2, figsize=(2 * size + 1, size))
    sns.heatmap(mean_abs.astype(float), ax=axes[0], cmap="viridis", vmin=0, vmax=1, annot=len(mean_abs) <= 12,
                fmt=".2f", square=True, cbar_kws={"label": "mean |rho|"})
    axes[0].set_title(f"Association: mean |rho| between each pair of {group_label} groups\n(diagonal = within one {group_label})")
    sns.heatmap(percent_redundant.astype(float), ax=axes[1], cmap="magma", vmin=0,
                annot=len(mean_abs) <= 12, fmt=".0f", square=True, cbar_kws={"label": "% of pairs"})
    axes[1].set_title(f"Redundancy: % of pairs with |rho| >= {threshold}")
    for ax in axes:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45)
        plt.setp(ax.get_xticklabels(), ha="right")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_redundancy_overview(rho, summary, title, threshold):
    """|rho| distribution within vs between classes, and features kept per class after redundancy removal."""
    info = describe_features(rho.columns)
    classes = info["feature_class"].to_numpy()
    upper = np.triu(np.ones(rho.shape, dtype=bool), k=1)
    same = classes[:, None] == classes[None, :]
    absolute = rho.abs().to_numpy(copy=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    bins = np.linspace(0, 1, 41)
    axes[0].hist(absolute[upper & same], bins=bins, density=True, alpha=0.6, label="same class")
    axes[0].hist(absolute[upper & ~same], bins=bins, density=True, alpha=0.6, label="different classes")
    axes[0].axvline(threshold, color="k", ls="--", lw=1, label=f"threshold {threshold}")
    axes[0].set_xlabel("|Spearman rho|")
    axes[0].set_ylabel("density of feature pairs")
    axes[0].set_title("Correlation within vs between feature classes")
    axes[0].legend()

    x = np.arange(len(summary))
    axes[1].bar(x, summary["features"], color="lightgrey", label="all features")
    axes[1].bar(x, summary["kept"], color=[class_palette(summary.index)[c] for c in summary.index],
                label="kept (one per redundant group)")
    for i, (total, pct) in enumerate(zip(summary["features"], summary["percent_redundant"])):
        axes[1].text(i, total, f"{pct:.0f}% redundant", ha="center", va="bottom", fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(summary.index, rotation=30)
    axes[1].set_ylabel("number of features")
    axes[1].set_title(f"Features per class before / after removing redundancy (|rho| >= {threshold})")
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig


#whole analysis
def run_redundancy_analysis(csv_path, output_dir, threshold=0.9, channel=None):
    """Run every step for one feature CSV, save tables and figures to <output_dir>/<csv name>/, return the results."""
    csv_path = Path(csv_path)
    name = csv_path.stem + ("" if channel is None else f"_ch{channel}")
    out = Path(output_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    features, constant = load_feature_matrix(csv_path, channel)
    rho = spearman_matrix(features)
    pairs = redundant_pairs(rho, threshold)
    clusters = redundancy_clusters(rho, threshold)
    info = describe_features(rho.columns)
    class_mean, class_percent = group_association(rho, info["feature_class"], threshold, CLASS_ORDER)
    summary = class_summary(clusters)

    results = {"features": features, "constant_features": constant, "rho": rho, "redundant_pairs": pairs,
               "clusters": clusters, "class_association": class_mean, "class_redundancy": class_percent,
               "class_summary": summary, "figures": {}}
    results["figures"]["clustermap"] = plot_correlation_clustermap(rho, name)
    results["figures"]["class_association"] = plot_group_association(class_mean, class_percent, name,
                                                                     "feature class", threshold)
    results["figures"]["overview"] = plot_redundancy_overview(rho, summary, name, threshold)
    if info["image_type"].nunique() > 1:
        type_mean, type_percent = group_association(rho, info["image_type"], threshold)
        results["image_type_association"], results["image_type_redundancy"] = type_mean, type_percent
        results["figures"]["image_type_association"] = plot_group_association(type_mean, type_percent, name,
                                                                              "image type", threshold)

    # save tables and figures
    rho.to_csv(out / "spearman_matrix.csv")
    pairs.to_csv(out / "redundant_pairs.csv", index=False)
    clusters.to_csv(out / "redundancy_clusters.csv", index=False)
    clusters.loc[clusters["representative"], "feature"].to_csv(out / "non_redundant_features.csv", index=False)
    class_mean.to_csv(out / "class_association_mean_abs_rho.csv")
    class_percent.to_csv(out / "class_redundancy_percent_pairs.csv")
    summary.to_csv(out / "class_summary.csv")
    if "image_type_association" in results:
        results["image_type_association"].to_csv(out / "image_type_association_mean_abs_rho.csv")
        results["image_type_redundancy"].to_csv(out / "image_type_redundancy_percent_pairs.csv")
    for fig_name, fig in results["figures"].items():
        fig.savefig(out / f"{fig_name}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)                      # shown later with display(); avoids showing every figure twice

    kept = int(clusters["representative"].sum())
    print(f"{name}: {len(features)} samples, {features.shape[1]} features ({len(constant)} constant dropped) | "
          f"{len(pairs)} redundant pairs (|rho| >= {threshold}) | {kept} non-redundant features kept | saved to {out}")
    return results
