"""Select the final features of a feature CSV: keep features that are reproducible and not redundant.

Step 1, reproducibility: a feature is kept if its ICC(2,1) (from outputs/icc/<csv name>/icc_per_channel.csv, made by
         ICC_check.ipynb) is >= icc_threshold in every MRI channel.
Step 2, redundancy: of the reproducible features, the most reproducible one is kept first; every next feature is
         kept only if its |Spearman rho| with all features kept so far is < redundancy_threshold.

Output: a CSV with the identifier columns and the selected features, and a report with the decision per feature.
"""
from pathlib import Path

import pandas as pd

ID_COLUMNS = ["case", "side", "channel", "task"]


def load_icc(icc_csv):
    """ICC(2,1) of every feature: the lowest value over the MRI channels (shape features have one value)."""
    icc = pd.read_csv(icc_csv)
    icc["feature"] = icc["image_type"] + "_" + icc["feature_class"] + "_" + icc["feature_name"]
    column = "ICC(2,1)" if "ICC(2,1)" in icc.columns else "icc2_1"      # ICC_check.ipynb tables use "ICC(2,1)"
    return icc.groupby("feature")[column].min()      # use .median() instead to be less strict


def select_final_features(csv_path, icc_csv, output_dir, icc_threshold=0.99, redundancy_threshold=0.99,
                          keep_untested=False):
    """Write <output_dir>/<csv name> with the selected features and <output_dir>/selection_<csv name>.

    keep_untested: what to do with features that have no ICC (e.g. Gabor and Laws, which were not part of the
    reproducibility run). False = drop them, True = keep them and only check their redundancy.
    """
    csv_path = Path(csv_path)
    table = pd.read_csv(csv_path)
    ids = [c for c in ID_COLUMNS if c in table.columns]
    features = table.drop(columns=ids)
    icc = load_icc(icc_csv)

    report = pd.DataFrame({"feature": features.columns})
    report["icc2_1"] = report["feature"].map(icc)
    report["constant"] = [features[c].nunique() <= 1 for c in features.columns]

    # step 1: reproducibility
    report["reproducible"] = report["icc2_1"] >= icc_threshold
    if keep_untested:
        report["reproducible"] |= report["icc2_1"].isna()
    report["reproducible"] &= ~report["constant"]          # constant features carry no information

    # step 2: redundancy, most reproducible features first
    candidates = report[report["reproducible"]].sort_values("icc2_1", ascending=False)["feature"].tolist()
    rho = features[candidates].corr(method="spearman").abs()
    kept, redundant_with = [], {}
    for feature in candidates:
        too_similar = [k for k in kept if rho.loc[feature, k] >= redundancy_threshold]
        if too_similar:
            redundant_with[feature] = too_similar[0]
        else:
            kept.append(feature)
    report["redundant_with"] = report["feature"].map(redundant_with)
    report["selected"] = report["feature"].isin(kept)

    # save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_columns = [c for c in features.columns if c in kept]           # original column order
    table[ids + selected_columns].to_csv(output_dir / csv_path.name, index=False)
    report.to_csv(output_dir / f"selection_{csv_path.name}", index=False)

    n_untested = int(report["icc2_1"].isna().sum())
    print(f"{csv_path.name}: {len(report)} features | {n_untested} without ICC | "
          f"{int(report['reproducible'].sum())} reproducible (ICC >= {icc_threshold}) | "
          f"{len(kept)} selected after removing redundancy (|rho| >= {redundancy_threshold})")
    return report
