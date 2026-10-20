"""Statistics and visualization of the final (reproducible, non-redundant) radiomic features.

Input: the CSVs in outputs/features/final (made by feature_matrix.ipynb), one per filter, with the identifier
columns case / side / channel / task and one column per feature, named '<image type>_<class>_<feature>'.

Study design of the rows: every case (patient) has two sides (L / R breast) and every side four MRI channels,
so rows are NOT independent. The tests below respect this:

    channel   repeated measures inside a breast  -> Friedman test, blocks = case + side (Kendall's W)
    side      paired inside a case and channel    -> Wilcoxon signed-rank test (matched-pairs rank-biserial r)
    outcome   independent groups (e.g. a label you add later) -> Kruskal-Wallis (epsilon^2) / Mann-Whitney (r)

p-values are corrected for the number of features with Benjamini-Hochberg (q-values).

Multivariate: PCA, PERMANOVA (permutations restricted inside blocks), a dispersion check (PERMDISP-like), and a
cross-validated classifier (leave-one-patient-out) with a permutation test.

Relations: mean |Spearman rho| and the modified RV coefficient between feature classes and between filters,
and a variance decomposition of every feature into patient / side / channel / residual.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, permutation_test_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ID_COLUMNS = ["case", "side", "channel", "task"]
CLASS_ORDER = ["shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]
FILTER_ORDER = ["raw_image", "square", "square_root", "logarithmic", "exponential", "gradient",
                "log", "wavelet", "laws", "gabor", "binary_patterns"]

# colors: fixed categorical order, one-hue sequential ramp, blue <-> red diverging with a gray midpoint
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CLASS_COLORS = dict(zip(CLASS_ORDER, PALETTE))
SURFACE, INK, INK_2, MUTED, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5",
                                                        "#1c5cab", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("blue_red", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a32b2b"])


# ----------------------------------------------------------------------------------------------- loading
def filter_name(csv_path):
    """'features_square_root_filter.csv' -> 'square_root'."""
    return Path(csv_path).stem.replace("features_", "").replace("_filter", "")


def describe_features(columns, filter=None):
    """Table with filter, image type, class and name of every feature column."""
    parts = [c.split("_", 2) for c in columns]
    return pd.DataFrame({"feature": list(columns), "filter": filter,
                         "image_type": [p[0] for p in parts], "feature_class": [p[1] for p in parts],
                         "feature_name": [p[2] for p in parts]})


def load_final_features(feature_dir):
    """Merge all final feature CSVs of feature_dir into one table.

    Returns (table, info): table has the identifier columns and every feature, info has one row per feature
    (feature, filter, image_type, feature_class, feature_name).
    """
    csvs = sorted(Path(feature_dir).glob("features_*.csv"),
                  key=lambda p: FILTER_ORDER.index(filter_name(p)) if filter_name(p) in FILTER_ORDER else 99)
    table, info = None, []
    for csv_path in csvs:
        part = pd.read_csv(csv_path)
        columns = [c for c in part.columns if c not in ID_COLUMNS]
        info.append(describe_features(columns, filter_name(csv_path)))
        if table is None:
            table = part
        else:
            duplicated = set(columns) & set(table.columns)
            if duplicated:
                raise ValueError(f"{csv_path.name}: columns already loaded: {sorted(duplicated)[:5]}")
            table = table.merge(part.drop(columns="task"), on=["case", "side", "channel"], how="inner")
    info = pd.concat(info, ignore_index=True)
    print(f"{len(csvs)} filters | {len(table)} samples | {len(info)} features")
    return table, info


def feature_columns(table):
    """All columns except the identifiers."""
    return [c for c in table.columns if c not in ID_COLUMNS]


def short_name(feature):
    """Shorter feature label for plots, e.g. 'gabor-sigma-2-0-mm-lambda-4-0-mm-axial_glcm_Idmn' -> 'gabor-s2-l4-axial_glcm_Idmn'."""
    return (feature.replace("sigma-", "s").replace("lambda-", "l").replace("-0-mm", "")
            .replace("-energy", "").replace("original_", "raw_"))


def standardize(table, features):
    """z-scores (mean 0, sd 1) of the features; constant features are dropped."""
    values = table[features].astype(float)
    values = values.loc[:, values.std() > 0]
    return (values - values.mean()) / values.std()


def selection_counts(feature_dir):
    """Number of extracted, reproducible and selected features per filter (from the selection_*.csv reports)."""
    rows = []
    for report_path in sorted(Path(feature_dir).glob("selection_features_*.csv")):
        report = pd.read_csv(report_path)
        rows.append({"filter": filter_name(report_path.name.replace("selection_", "")),
                     "extracted": len(report), "reproducible": int(report["reproducible"].sum()),
                     "selected": int(report["selected"].sum())})
    counts = pd.DataFrame(rows).set_index("filter")
    counts = counts.loc[[f for f in FILTER_ORDER if f in counts.index]]
    counts["retention (%)"] = (100 * counts["selected"] / counts["extracted"]).round(1)
    return counts


def count_by_filter_and_class(info):
    """Number of final features per filter (rows) and feature class (columns)."""
    counts = pd.crosstab(info["filter"], info["feature_class"])
    return counts.reindex(index=[f for f in FILTER_ORDER if f in counts.index],
                          columns=[c for c in CLASS_ORDER if c in counts.columns], fill_value=0)


# ---------------------------------------------------------------------------------- descriptive statistics
def descriptive_statistics(table, info):
    """Per feature: n, mean, sd, median, quartiles, IQR, min, max, coefficient of variation, skewness,
    excess kurtosis and the Shapiro-Wilk normality test (normal = p >= 0.05)."""
    values = table[info["feature"]].astype(float)
    desc = info.copy()
    desc["n"] = values.count().to_numpy()
    desc["mean"] = values.mean().to_numpy()
    desc["sd"] = values.std().to_numpy()
    desc["median"] = values.median().to_numpy()
    desc["q1"] = values.quantile(0.25).to_numpy()
    desc["q3"] = values.quantile(0.75).to_numpy()
    desc["iqr"] = desc["q3"] - desc["q1"]
    desc["min"] = values.min().to_numpy()
    desc["max"] = values.max().to_numpy()
    desc["cv"] = desc["sd"] / desc["mean"].abs().replace(0, np.nan)
    desc["skewness"] = stats.skew(values, nan_policy="omit")
    desc["kurtosis"] = stats.kurtosis(values, nan_policy="omit")
    desc["shapiro_p"] = [stats.shapiro(values[c].dropna()).pvalue if values[c].nunique() > 2 else np.nan
                         for c in values.columns]
    desc["normal"] = desc["shapiro_p"] >= 0.05
    return desc


def descriptive_by_group(table, features, group):
    """Long table: n, mean, sd, median and IQR of every feature in every group (e.g. channel)."""
    long = table.melt(id_vars=[group], value_vars=list(features), var_name="feature")
    summary = long.groupby(["feature", group])["value"].agg(
        n="count", mean="mean", sd="std", median="median",
        iqr=lambda v: v.quantile(0.75) - v.quantile(0.25))
    return summary.reset_index()


def normality_summary(desc, by="filter"):
    """Per filter (or class): number of features, % normally distributed (Shapiro-Wilk) and median |skewness|."""
    summary = desc.groupby(by).agg(features=("feature", "size"), normal_percent=("normal", "mean"),
                                   median_abs_skewness=("skewness", lambda s: s.abs().median()),
                                   median_cv=("cv", "median"))
    summary["normal_percent"] = (100 * summary["normal_percent"]).round(1)
    order = FILTER_ORDER if by == "filter" else CLASS_ORDER
    return summary.loc[[o for o in order if o in summary.index]]


# --------------------------------------------------------------------------------------- univariate tests
def add_fdr(results, p_column="p"):
    """Add the Benjamini-Hochberg q-value of every test (NaN p-values are left out)."""
    results = results.copy()
    valid = results[p_column].notna()
    results["q"] = np.nan
    if valid.any():
        results.loc[valid, "q"] = stats.false_discovery_control(results.loc[valid, p_column].to_numpy())
    return results


def _blocks(table, block):
    """One label per block, e.g. block = ['case', 'side'] -> 'BreastMRI002_L'."""
    return table[block].astype(str).agg("_".join, axis=1)


def repeated_measures_test(table, info, group, block, alpha=0.05):
    """Test every feature for a difference between the levels of `group` measured inside the same block.

    more than 2 levels: Friedman test, effect size Kendall's W (0 = no agreement, 1 = same order in every block).
    2 levels: Wilcoxon signed-rank test, effect size matched-pairs rank-biserial r (-1..1, sign = direction).
    Features without variation inside the blocks (e.g. shape features over channels) get p = NaN.
    """
    blocks = _blocks(table, block)
    levels = sorted(table[group].unique())
    rows = []
    for feature in info["feature"]:
        wide = pd.DataFrame({"block": blocks, "group": table[group], "value": table[feature]}) \
            .pivot_table(index="block", columns="group", values="value").dropna()
        wide = wide[levels]
        row = {"feature": feature, "n_blocks": len(wide), "statistic": np.nan, "p": np.nan, "effect": np.nan}
        if (wide.nunique(axis=1) > 1).any():
            if len(levels) > 2:
                result = stats.friedmanchisquare(*[wide[level] for level in levels])
                row["statistic"], row["p"] = result.statistic, result.pvalue
                row["effect"] = result.statistic / (len(wide) * (len(levels) - 1))       # Kendall's W
            else:
                difference = wide[levels[0]] - wide[levels[1]]
                difference = difference[difference != 0]
                result = stats.wilcoxon(difference)
                ranks = stats.rankdata(difference.abs())
                row["statistic"], row["p"] = result.statistic, result.pvalue
                row["effect"] = (ranks[difference > 0].sum() - ranks[difference < 0].sum()) / ranks.sum()
            row.update({f"median_{level}": wide[level].median() for level in levels})
        rows.append(row)
    results = add_fdr(info.merge(pd.DataFrame(rows), on="feature"))
    results["significant"] = results["q"] < alpha
    results["test"] = "Friedman" if len(levels) > 2 else "Wilcoxon signed-rank"
    results["effect_size"] = "Kendall W" if len(levels) > 2 else "rank-biserial r"
    print(f"{group}: {int(results['significant'].sum())} of {results['p'].notna().sum()} tested features "
          f"differ (q < {alpha})")
    return results.sort_values("p").reset_index(drop=True)


def independent_groups_test(table, info, group, alpha=0.05):
    """Test every feature for a difference between independent groups (e.g. a diagnosis label, one row per patient).

    more than 2 groups: Kruskal-Wallis test, effect size epsilon^2 (0..1).
    2 groups: Mann-Whitney U test, effect size rank-biserial r (-1..1, sign = direction).
    """
    levels = sorted(table[group].dropna().unique())
    n = int(table[group].notna().sum())
    rows = []
    for feature in info["feature"]:
        samples = [table.loc[table[group] == level, feature].dropna() for level in levels]
        row = {"feature": feature, "statistic": np.nan, "p": np.nan, "effect": np.nan}
        if pd.concat(samples).nunique() > 1:
            if len(levels) > 2:
                result = stats.kruskal(*samples)
                row["effect"] = result.statistic / (n - 1)                                  # epsilon^2
            else:
                result = stats.mannwhitneyu(*samples, alternative="two-sided")
                row["effect"] = 2 * result.statistic / (len(samples[0]) * len(samples[1])) - 1
            row["statistic"], row["p"] = result.statistic, result.pvalue
        row.update({f"median_{level}": s.median() for level, s in zip(levels, samples)})
        rows.append(row)
    results = add_fdr(info.merge(pd.DataFrame(rows), on="feature"))
    results["significant"] = results["q"] < alpha
    results["test"] = "Kruskal-Wallis" if len(levels) > 2 else "Mann-Whitney U"
    print(f"{group}: {int(results['significant'].sum())} of {results['p'].notna().sum()} tested features "
          f"differ (q < {alpha})")
    return results.sort_values("p").reset_index(drop=True)


def pairwise_posthoc(table, feature, group, block):
    """Post-hoc test of one feature: Wilcoxon signed-rank test between every pair of levels, Holm corrected."""
    wide = pd.DataFrame({"block": _blocks(table, block), "group": table[group], "value": table[feature]}) \
        .pivot_table(index="block", columns="group", values="value").dropna()
    levels = list(wide.columns)
    rows = []
    for i, a in enumerate(levels):
        for b in levels[i + 1:]:
            result = stats.wilcoxon(wide[a], wide[b])
            rows.append({"feature": feature, "level_1": a, "level_2": b,
                         "median_difference": (wide[a] - wide[b]).median(), "p": result.pvalue})
    posthoc = pd.DataFrame(rows)
    order = np.argsort(posthoc["p"].to_numpy())                         # Holm step-down correction
    holm = np.maximum.accumulate((len(order) - np.arange(len(order))) * posthoc["p"].to_numpy()[order])
    posthoc.loc[posthoc.index[order], "p_holm"] = np.minimum(holm, 1)
    return posthoc


def significance_summary(results, by="filter"):
    """Per filter (or class): number of tested features, number and % significant, median effect size."""
    tested = results[results["p"].notna()]
    summary = tested.groupby(by).agg(tested=("feature", "size"), significant=("significant", "sum"),
                                     median_effect=("effect", lambda e: e.abs().median()))
    summary["significant_percent"] = (100 * summary["significant"] / summary["tested"]).round(1)
    order = FILTER_ORDER if by == "filter" else CLASS_ORDER
    return summary.loc[[o for o in order if o in summary.index]]


def variance_components(table, info, case="case", side="side", channel="channel"):
    """Share of the total variance of every feature explained by patient, side (inside patient) and channel.

    Uses sums of squares of the balanced design case x side x channel; the residual is the side x channel
    interaction. The four shares add up to 1.
    """
    rows = []
    breast = _blocks(table, [case, side])
    for feature in info["feature"]:
        y = table[feature].astype(float)
        total = ((y - y.mean()) ** 2).sum()
        if total == 0:
            continue
        ss_case = ((y.groupby(table[case]).transform("mean") - y.mean()) ** 2).sum()
        ss_breast = ((y.groupby(breast).transform("mean") - y.mean()) ** 2).sum()
        ss_channel = ((y.groupby(table[channel]).transform("mean") - y.mean()) ** 2).sum()
        rows.append({"feature": feature, "patient": ss_case / total, "side": (ss_breast - ss_case) / total,
                     "channel": ss_channel / total,
                     "residual": max(total - ss_breast - ss_channel, 0) / total})
    return info.merge(pd.DataFrame(rows), on="feature")


# ------------------------------------------------------------------------------- association between groups
def mean_abs_spearman(table, info, by="feature_class"):
    """Mean |Spearman rho| between all features of two groups (feature classes or filters).

    Diagonal: mean |rho| between different features of the same group (internal redundancy).
    """
    rho = table[info["feature"]].corr(method="spearman").abs().to_numpy(copy=True)
    np.fill_diagonal(rho, np.nan)
    labels = info[by].to_numpy()
    order = [o for o in (CLASS_ORDER if by == "feature_class" else FILTER_ORDER) if o in set(labels)]
    matrix = pd.DataFrame(index=order, columns=order, dtype=float)
    for a in order:
        for b in order:
            matrix.loc[a, b] = np.nanmean(rho[np.ix_(labels == a, labels == b)])
    return matrix


def rv_coefficient(x, y):
    """Modified RV coefficient (Smilde et al. 2009) between two standardized feature blocks with the same rows.

    0 = the blocks describe unrelated patterns over the samples, 1 = the same pattern. The modified version
    removes the diagonal of the sample configuration matrices, which avoids the upward bias of the classic RV
    when there are many features and few samples.
    """
    sx, sy = x @ x.T, y @ y.T
    np.fill_diagonal(sx, 0)
    np.fill_diagonal(sy, 0)
    return float((sx * sy).sum() / np.sqrt((sx * sx).sum() * (sy * sy).sum()))


def rv_matrix(table, info, by="filter"):
    """Modified RV coefficient between every pair of filters (or feature classes)."""
    z = standardize(table, info["feature"])
    labels = info.set_index("feature").loc[z.columns, by].to_numpy()
    order = [o for o in (CLASS_ORDER if by == "feature_class" else FILTER_ORDER) if o in set(labels)]
    blocks = {o: z.loc[:, labels == o].to_numpy() for o in order}
    return pd.DataFrame([[rv_coefficient(blocks[a], blocks[b]) for b in order] for a in order],
                        index=order, columns=order)


def top_cross_pairs(table, info, by="filter", n=20):
    """The n most strongly correlated feature pairs whose features belong to different filters (or classes)."""
    features = info["feature"].to_numpy()
    rho = table[features].corr(method="spearman").to_numpy()
    labels = info[by].to_numpy()
    i, j = np.triu_indices(len(features), k=1)
    keep = labels[i] != labels[j]
    i, j = i[keep], j[keep]
    best = np.argsort(-np.abs(rho[i, j]))[:n]
    return pd.DataFrame({"feature_1": features[i[best]], f"{by}_1": labels[i[best]],
                         "feature_2": features[j[best]], f"{by}_2": labels[j[best]],
                         "rho": rho[i[best], j[best]]})


# ------------------------------------------------------------------------------------------- multivariate
def run_pca(table, info, n_components=10):
    """PCA of the standardized features. Returns (scores with identifier columns, explained variance, loadings)."""
    z = standardize(table, info["feature"])
    pca = PCA(n_components=min(n_components, len(z) - 1)).fit(z)
    names = [f"PC{i + 1}" for i in range(pca.n_components_)]
    scores = pd.concat([table[ID_COLUMNS].reset_index(drop=True),
                        pd.DataFrame(pca.transform(z), columns=names)], axis=1)
    explained = pd.DataFrame({"component": names, "explained_percent": 100 * pca.explained_variance_ratio_})
    explained["cumulative_percent"] = explained["explained_percent"].cumsum()
    loadings = pd.DataFrame(pca.components_.T, index=z.columns, columns=names)
    loadings = info.set_index("feature").loc[z.columns].join(loadings).rename_axis("feature").reset_index()
    return scores, explained, loadings


def _pseudo_f(x, labels):
    """PERMANOVA pseudo-F and R^2 for Euclidean distances (computed from sums of squares)."""
    levels = np.unique(labels)
    ss_total = ((x - x.mean(axis=0)) ** 2).sum()
    ss_within = sum(((x[labels == g] - x[labels == g].mean(axis=0)) ** 2).sum() for g in levels)
    k, n = len(levels), len(x)
    return ((ss_total - ss_within) / (k - 1)) / (ss_within / (n - k)), (ss_total - ss_within) / ss_total


def permanova(table, info, group, strata=None, n_perm=999, seed=0):
    """PERMANOVA (Anderson 2001) on the Euclidean distance of the standardized features.

    Tests if the multivariate centroids of the groups differ. strata: columns inside which labels are permuted
    (e.g. ['case', 'side'] when testing channel), so that the repeated-measures design is respected.
    """
    x = standardize(table, info["feature"]).to_numpy()
    labels = table[group].astype(str).to_numpy()
    f_obs, r2 = _pseudo_f(x, labels)
    rng = np.random.default_rng(seed)
    strata_index = [np.arange(len(labels))] if strata is None else \
        [np.flatnonzero(_blocks(table, strata).to_numpy() == s) for s in _blocks(table, strata).unique()]
    f_perm = np.empty(n_perm)
    for b in range(n_perm):
        permuted = labels.copy()
        for index in strata_index:
            permuted[index] = rng.permutation(labels[index])
        f_perm[b] = _pseudo_f(x, permuted)[0]
    p = (1 + (f_perm >= f_obs).sum()) / (n_perm + 1)
    return pd.Series({"group": group, "strata": strata, "pseudo_F": f_obs, "R2": r2, "p": p,
                      "permutations": n_perm})


def dispersion_test(table, info, group):
    """PERMDISP-like check: distance of every sample to its group centroid, compared with Kruskal-Wallis.

    A significant PERMANOVA with a non-significant dispersion test means the groups differ in location
    (centroid), not just in spread.
    """
    z = standardize(table, info["feature"])
    labels = table[group].to_numpy()
    distance = np.zeros(len(z))
    for g in np.unique(labels):
        members = z[labels == g]
        distance[labels == g] = np.sqrt(((members - members.mean()) ** 2).sum(axis=1))
    result = stats.kruskal(*[distance[labels == g] for g in np.unique(labels)])
    per_group = pd.Series(distance).groupby(labels).mean().round(2).to_dict()
    return pd.Series({"group": group, "mean_distance_to_centroid": per_group, "H": result.statistic,
                      "p": result.pvalue})


def classification_cv(table, info, target, cv_groups="case", n_perm=200, seed=0):
    """How well can all features together predict `target`? Cross-validated multivariate analysis.

    L2 logistic regression on standardized features, leave-one-patient-out cross-validation (no patient is in
    training and test at the same time), balanced accuracy, and a permutation test (labels permuted inside each
    patient) for its p-value. Returns (summary, permutation scores).
    """
    x = table[info["feature"]].astype(float).to_numpy()
    y = table[target].astype(str).to_numpy()
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))
    score, perm_scores, p = permutation_test_score(
        model, x, y, groups=table[cv_groups].to_numpy(), cv=LeaveOneGroupOut(), scoring="balanced_accuracy",
        n_permutations=n_perm, random_state=seed, n_jobs=-1)
    summary = pd.Series({"target": target, "balanced_accuracy": score, "chance_level": 1 / len(np.unique(y)),
                         "permutation_mean": perm_scores.mean(), "p": p, "permutations": n_perm})
    return summary, perm_scores


# ------------------------------------------------------------------------------------------ visualization
def set_plot_style():
    """Quiet axes and grid, readable text, white surface."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": NEUTRAL, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "text.color": INK, "legend.frameon": False, "legend.fontsize": 8, "font.size": 9,
        "lines.linewidth": 2, "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight"})


def save_figure(fig, path):
    """Save a figure as PNG (folders are created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"saved {path}")


def group_colors(levels):
    """Fixed categorical color for every level, in sorted order."""
    return {level: PALETTE[i % len(PALETTE)] for i, level in enumerate(sorted(levels))}


def _heatmap(ax, matrix, title, cmap=SEQUENTIAL, vmin=None, vmax=None, fmt=".2f", label=""):
    sns.heatmap(matrix.astype(float), ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, annot=True, fmt=fmt,
                annot_kws={"size": 7}, linewidths=2, linecolor=SURFACE, square=True,
                cbar_kws={"shrink": 0.7, "label": label})
    ax.set_title(title, loc="left")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    ax.tick_params(length=0)


def plot_selection_funnel(counts):
    """Extracted, reproducible and selected features per filter, plus the % kept."""
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(counts))
    for i, (column, color) in enumerate(zip(["extracted", "reproducible", "selected"],
                                            ["#86b6ef", "#2a78d6", "#104281"])):
        ax.barh(y + (i - 1) * 0.27, counts[column], height=0.25, color=color, label=column)
    for yi, (n, pct) in enumerate(zip(counts["extracted"], counts["retention (%)"])):
        ax.text(n + 8, yi - 0.27, f"{pct:.0f}% kept", va="center", fontsize=7, color=INK_2)
    ax.set_yticks(y, counts.index)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("number of features")
    ax.set_title("Features per filter: extracted → reproducible → selected", loc="left")
    ax.legend(loc="lower right")
    return fig


def plot_count_heatmap(counts, title="Final features per filter and feature class"):
    """Heatmap of counts (rows = filters, columns = feature classes)."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    _heatmap(ax, counts, title, fmt=".0f", label="features")
    return fig


def plot_descriptive_overview(desc):
    """Per filter: % of features that are normally distributed, and the spread of |skewness| and CV."""
    order = [f for f in FILTER_ORDER if f in set(desc["filter"])]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    normal = (100 * desc.groupby("filter")["normal"].mean().loc[order]).rename("percent").reset_index()
    sns.barplot(data=normal, y="filter", x="percent", order=order, ax=axes[0], color=PALETTE[0], width=0.6)
    axes[0].set_ylabel("")
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("% features normal (Shapiro-Wilk p ≥ 0.05)")
    axes[0].set_title("Normality", loc="left")
    for ax, column, label in [(axes[1], "skewness", "|skewness|"), (axes[2], "cv", "|coefficient of variation|")]:
        data = desc.assign(value=desc[column].abs())
        sns.boxplot(data=data, y="filter", x="value", order=order, ax=ax, color="#86b6ef", fliersize=2,
                    linewidth=1, width=0.6)
        ax.set_xscale("log")
        ax.set_xlabel(label + " (log scale)")
        ax.set_ylabel("")
        ax.set_title(label.strip("|").capitalize(), loc="left")
    for ax in axes:
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_volcano(results, alpha=0.05, n_labels=6, title=""):
    """Effect size against -log10(q) of every feature, colored by feature class; the strongest are labeled."""
    data = results[results["p"].notna()].copy()
    data["minus_log_q"] = -np.log10(data["q"].clip(lower=1e-300))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for feature_class in [c for c in CLASS_ORDER if c in set(data["feature_class"])]:
        part = data[data["feature_class"] == feature_class]
        ax.scatter(part["effect"], part["minus_log_q"], s=18, color=CLASS_COLORS[feature_class], alpha=0.75,
                   edgecolor=SURFACE, linewidth=0.5, label=feature_class)
    ax.axhline(-np.log10(alpha), color=MUTED, linestyle="--", linewidth=1)
    ax.text(ax.get_xlim()[0], -np.log10(alpha), f" q = {alpha}", va="bottom", fontsize=7, color=MUTED)
    top = data.reindex(data["effect"].abs().sort_values(ascending=False).index).head(n_labels)
    for i, (_, row) in enumerate(top.iterrows()):          # labels stacked below each other, left of the points
        ax.annotate(short_name(row["feature"]), (row["effect"], row["minus_log_q"]), fontsize=6.5, color=INK_2,
                    xytext=(-40 if row["effect"] > 0 else 40, -14 * (i + 1)), textcoords="offset points",
                    ha="right" if row["effect"] > 0 else "left", va="center",
                    bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": "none", "alpha": 0.85},
                    arrowprops={"arrowstyle": "-", "color": NEUTRAL, "linewidth": 0.6})
    ax.set_xlabel(f"effect size ({data['effect_size'].iloc[0] if 'effect_size' in data else 'effect'})")
    ax.set_ylabel("-log10(q)")
    ax.set_title(title or f"{data['test'].iloc[0]} test of every feature", loc="left")
    ax.legend(title="feature class", loc="upper left", bbox_to_anchor=(1, 1))
    return fig


def plot_significance_heatmap(results, title="% of features with q < 0.05"):
    """Heatmap of the % significant features per filter (rows) and feature class (columns)."""
    tested = results[results["p"].notna()]
    matrix = 100 * tested.pivot_table(index="filter", columns="feature_class", values="significant",
                                      aggfunc="mean")
    matrix = matrix.reindex(index=[f for f in FILTER_ORDER if f in matrix.index],
                            columns=[c for c in CLASS_ORDER if c in matrix.columns])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    _heatmap(ax, matrix, title, vmin=0, vmax=100, fmt=".0f", label="% significant")
    return fig


def plot_top_features(table, results, group, block=None, n=6, ncols=3):
    """Box plots of the n most significant features per group level; lines connect the same block."""
    top = results[results["p"].notna()].head(n)
    levels = sorted(table[group].unique())
    colors = group_colors(levels)
    nrows = int(np.ceil(len(top) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.3 * nrows), squeeze=False)
    for ax, (_, row) in zip(axes.flat, top.iterrows()):
        data = table[[group, row["feature"]] + (block or [])].copy()
        data["z"] = (data[row["feature"]] - data[row["feature"]].mean()) / data[row["feature"]].std()
        if block:
            data["block"] = _blocks(data, block)
            for _, members in data.groupby("block"):
                members = members.sort_values(group)
                ax.plot([levels.index(v) for v in members[group]], members["z"], color=GRID, linewidth=0.8,
                        zorder=1)
        sns.boxplot(data=data, x=group, y="z", order=levels, hue=group, hue_order=levels, palette=colors,
                    ax=ax, width=0.5, fliersize=0, linewidth=1, boxprops={"alpha": 0.35}, legend=False)
        sns.stripplot(data=data, x=group, y="z", order=levels, hue=group, hue_order=levels, palette=colors,
                      ax=ax, size=4, jitter=0.08, edgecolor=SURFACE, linewidth=0.5, legend=False, zorder=3)
        ax.set_title(f"{short_name(row['feature'])}\nq = {row['q']:.1e}, effect = {row['effect']:.2f}",
                     loc="left", fontsize=8)
        ax.set_ylabel("z-score")
        ax.grid(axis="x", visible=False)
    for ax in list(axes.flat)[len(top):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_variance_components(vc, by="filter"):
    """Stacked bars: mean share of variance explained by patient, side, channel and residual per filter/class."""
    parts = ["patient", "side", "channel", "residual"]
    colors = [PALETTE[0], PALETTE[1], PALETTE[2], NEUTRAL]
    order = [o for o in (FILTER_ORDER if by == "filter" else CLASS_ORDER) if o in set(vc[by])]
    shares = 100 * vc.groupby(by)[parts].mean().loc[order]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(order) + 1.5))
    left = np.zeros(len(order))
    for part, color in zip(parts, colors):
        ax.barh(order, shares[part], left=left, color=color, edgecolor=SURFACE, linewidth=2, height=0.65,
                label=part)
        for yi, (value, start) in enumerate(zip(shares[part], left)):
            if value >= 8:
                ax.text(start + value / 2, yi, f"{value:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if part != "residual" else INK)
        left += shares[part].to_numpy()
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("mean % of feature variance")
    ax.set_title("Where does feature variance come from?", loc="left")
    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.06))
    return fig


def plot_association_heatmap(matrix, title, label="mean |Spearman rho|"):
    """Heatmap of an association matrix with values between 0 and 1."""
    fig, ax = plt.subplots(figsize=(0.6 * len(matrix) + 2.5, 0.55 * len(matrix) + 1.5))
    _heatmap(ax, matrix, title, vmin=0, vmax=1, label=label)
    return fig


def plot_clustermap(table, info, row_groups=("channel", "side")):
    """Hierarchically clustered heatmap of all samples (rows) x features (columns), z-scores clipped to +-3.

    Row color bars show the group of every sample, column color bars the feature class.
    """
    z = standardize(table, info["feature"]).clip(-3, 3)
    z.index = table["task"]
    # sample groups get blue / gray ramps so they do not clash with the feature-class colors of the columns
    ramps = [["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"], ["#c3c2b7", "#52514e", "#0b0b0b"]]
    row_palettes = {g: dict(zip(sorted(table[g].unique()), ramps[i % 2])) for i, g in enumerate(row_groups)}
    row_colors = pd.DataFrame({g: table[g].map(row_palettes[g]).to_numpy() for g in row_groups}, index=z.index)
    classes = info.set_index("feature").loc[z.columns, "feature_class"]
    col_colors = pd.Series(classes.map(CLASS_COLORS).to_numpy(), index=z.columns, name="class")
    grid = sns.clustermap(z, method="ward", metric="euclidean", cmap=DIVERGING, center=0, vmin=-3, vmax=3,
                          row_colors=row_colors, col_colors=col_colors, xticklabels=False,
                          yticklabels=True, figsize=(12, 9), dendrogram_ratio=(0.12, 0.08),
                          cbar_pos=(0.02, 0.86, 0.015, 0.1), cbar_kws={"ticks": [-3, 0, 3]})
    grid.ax_cbar.set_title("z-score", fontsize=7, loc="left")
    grid.ax_heatmap.set_xlabel(f"{z.shape[1]} features")
    grid.ax_heatmap.set_ylabel("")
    grid.ax_heatmap.tick_params(axis="y", labelsize=6)
    handles = [Patch(color=CLASS_COLORS[c], label=c) for c in CLASS_ORDER if c in set(classes)]
    for g in row_groups:
        handles += [Patch(color=color, label=f"{g} {level}") for level, color in row_palettes[g].items()]
    grid.ax_col_dendrogram.legend(handles=handles, ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                                  fontsize=7)
    return grid.fig


def plot_pca(scores, explained, color_by="channel", marker_by="side"):
    """Scree plot and PC1 vs PC2 score plot, colored by one grouping column and shaped by another."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.4]})
    ax = axes[0]
    ax.bar(explained["component"], explained["explained_percent"], color="#86b6ef", label="per component")
    ax.plot(explained["component"], explained["cumulative_percent"], color=PALETTE[0], marker="o",
            markersize=5, label="cumulative")
    ax.set_ylim(0, 100)
    ax.set_ylabel("% variance explained")
    ax.set_title("Scree plot", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="center right")
    ax = axes[1]
    colors = group_colors(scores[color_by].unique())
    markers = dict(zip(sorted(scores[marker_by].unique()), ["o", "s", "^", "D", "v", "P"]))
    for (c, m), part in scores.groupby([color_by, marker_by]):
        ax.scatter(part["PC1"], part["PC2"], s=55, color=colors[c], marker=markers[m], edgecolor=SURFACE,
                   linewidth=1.5)
    ax.set_xlabel(f"PC1 ({explained['explained_percent'].iloc[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained['explained_percent'].iloc[1]:.1f}%)")
    ax.set_title(f"PCA scores: color = {color_by}, shape = {marker_by}", loc="left")
    handles = [Line2D([], [], marker="o", linestyle="", color=col, label=f"{color_by} {lev}")
               for lev, col in colors.items()]
    handles += [Line2D([], [], marker=mk, linestyle="", color=MUTED, label=f"{marker_by} {lev}")
                for lev, mk in markers.items()]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    return fig


def plot_pca_loadings(loadings, component="PC1", n=15):
    """The n features with the largest |loading| on one component, colored by feature class."""
    top = loadings.reindex(loadings[component].abs().sort_values(ascending=False).index).head(n)
    fig, ax = plt.subplots(figsize=(8, 0.32 * n + 1.2))
    ax.barh([short_name(f) for f in top["feature"]], top[component], height=0.65,
            color=[CLASS_COLORS.get(c, NEUTRAL) for c in top["feature_class"]])
    ax.axvline(0, color=NEUTRAL, linewidth=1)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlabel(f"loading on {component}")
    ax.set_title(f"Top {n} features of {component}", loc="left")
    handles = [Patch(color=CLASS_COLORS[c], label=c) for c in CLASS_ORDER if c in set(top["feature_class"])]
    ax.legend(handles=handles, title="feature class", loc="upper left", bbox_to_anchor=(1, 1))
    return fig


def plot_permutation_null(summary, perm_scores):
    """Distribution of balanced accuracy under permuted labels, with the observed score and chance level."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(perm_scores, bins=20, color="#86b6ef", edgecolor=SURFACE, linewidth=1, label="permuted labels")
    ax.axvline(summary["balanced_accuracy"], color=PALETTE[1], linewidth=2,
               label=f"observed = {summary['balanced_accuracy']:.2f} (p = {summary['p']:.3f})")
    ax.axvline(summary["chance_level"], color=MUTED, linestyle="--", linewidth=1,
               label=f"chance = {summary['chance_level']:.2f}")
    ax.set_xlabel("balanced accuracy (leave-one-patient-out)")
    ax.set_ylabel("permutations")
    ax.set_title(f"Can all features predict {summary['target']}?", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    return fig
