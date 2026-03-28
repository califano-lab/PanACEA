"""
PanACEA: Pan-Cancer Assessment of Compound Effectors and Activity
Analysis pipeline for drug MoA and polypharmacology elucidation.

Authors: Lucas ZhongMing Hu et al.
"""

# ── Imports ───────────────────────────────────────────────────────────────────

import copy
import gc
import glob
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadr
import scipy.cluster.hierarchy
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── Constants ─────────────────────────────────────────────────────────────────

# Mapping of non-standard or duplicate drug names to their canonical forms.
DRUG_NAME_CORRECTIONS = {
    "Valproic": "Valproic acid",
    "Epothiloned": "Epothilone d",
    "Epothiloneb": "Epothilone b",
    "Cobimetinib (gdc0973, rg7420)": "Cobimetinib",
    "Bi2536": "Bi 2536",
    "Alectinib (ch5424802)x": "Alectinib",
    "(5z)7oxozeanol": "(5z)7oxozeaenol",
    "10debc": "10debc hydrochloride",
    "Ap26113": "Brigatinib",
    "Bi78d3": "Bi 78d3",
    "Bntx": "BNTX maleate salt hydrate",
    "Obatoclax": "Obatoclax mesylate",
    "Omacetaxine mepesuccinate": "Omacetaxine",
    "Nbenzylnaltrindole": "N benzylnaltrindole hydrochloride",
    "Buparlisib": "Bkm120",
    "Calmidazolium": "Calmidazolium chloride",
    "Cetylpyridium": "Cetylpyridinium chloride",
    "Dippa": "Dippa hydrochloride",
    "Fit": "Trifluoperazine",
    "Homoharringtonine": "Omacetaxine",
    "Idatraline": "Indatraline",
    "Lanatoside": "Lanatoside c",
    "Lde225": "Sonidegib",
    "Mycophenolate mofetil": "Mycophenolate",
    "Epx5676": "Epz5676",
    "Luminespib": "Auy922",
    "Cc292": "Avl292",
    "Serdemetan": "Jnj26854165",
    "Ipatasertib": "Gdc0068",
    "Silmitasertib": "Cx4945",
    "Patidegib": "Saridegib",
    "Plicamycin": "Mithramycin",
    "Vistusertib": "Azd2014",
    "Exherin": "Adh1",
    "Darolutamide": "Odm201",
    "Streptozocin": "Streptozotocin",
    "Voxtalisib": "Sar245409",
    "Navoximod": "Nlg919",
    "Tazemetostat": "Epz6438",
    "Pinometostat": "Epz5676",
    "Pictilisib": "Gdc0941",
    "Adavosertib": "Mk1775",
    "Midostaurin": "Pkc412",
    "Cddome": "Bardoxolone methyl",
    "Az12419304009": "Az12419304",
    "Az12456623010": "Az12456623",
    "Az12609721007": "Az12609721",
    "Az13064550016": "Az13064550",
    "Folinic acid": "Leucovorin",
    "Octreotide in water": "Octreotide",
}

# ── Drug name utilities ────────────────────────────────────────────────────────

def correct_drug_names(drug_name_list):
    """
    Standardize drug names using the canonical name mapping.

    Parameters
    ----------
    drug_name_list : list of str
        Raw drug names from the dataset.

    Returns
    -------
    list of str
        Corrected drug names.
    """
    corrected = []
    for name in drug_name_list:
        if name in DRUG_NAME_CORRECTIONS:
            canonical = DRUG_NAME_CORRECTIONS[name]
            print(f"  Renamed: '{name}' -> '{canonical}'")
            corrected.append(canonical)
        else:
            corrected.append(name)
    return corrected


def normalize_drug_name(raw_name):
    """
    Normalize a raw drug name by removing hyphens, slashes, and capitalizing.

    Parameters
    ----------
    raw_name : str

    Returns
    -------
    str
        Normalized drug name.
    """
    name = raw_name.replace("/", "-").replace("-", "")
    return name.capitalize() if name else name


# ── DataFrame helpers ──────────────────────────────────────────────────────────

def find_duplicate_indices(items):
    """
    Identify positions of duplicated entries in a list.

    Parameters
    ----------
    items : list

    Returns
    -------
    dict
        Maps each duplicated item to all its indices.
    list
        Flat list of every index involved in a duplication.
    """
    index_map = {}
    for idx, item in enumerate(items):
        index_map.setdefault(item, []).append(idx)

    duplicates = {k: v for k, v in index_map.items() if len(v) > 1}
    flat_indices = [i for indices in duplicates.values() for i in indices]
    return duplicates, flat_indices


def keep_closest_24h_and_highest_conc(df):
    """
    For each drug, retain the time point closest to 24 h and the highest
    concentration, then strip the trailing ``_<time>`` suffix from column names.

    Column name format expected: ``<drug>_<conc>_<time>``

    Parameters
    ----------
    df : pd.DataFrame
        Raw VIPER matrix with drug–concentration–time columns.

    Returns
    -------
    pd.DataFrame
        Filtered matrix with clean drug-name columns.
    """
    original_cols = df.columns.tolist()

    # ── Step 1: keep the time point closest to 24 h ──────────────────────────
    best_by_drug_conc = {}
    for col in original_cols:
        drug_conc = col[: col.rfind("_")]
        time = float(col.split("_")[-1])
        if drug_conc not in best_by_drug_conc:
            best_by_drug_conc[drug_conc] = col
        else:
            prev_time = float(best_by_drug_conc[drug_conc].split("_")[-1])
            if abs(time - 24) < abs(prev_time - 24):
                best_by_drug_conc[drug_conc] = col

    df = df[list(best_by_drug_conc.values())]

    # ── Step 2: keep the highest concentration per drug ───────────────────────
    best_by_drug = {}
    for col in df.columns.tolist():
        drug_conc = col[: col.rfind("_")]
        drug = "_".join(drug_conc.split("_")[:-1])
        conc = drug_conc.split("_")[-1]
        if drug not in best_by_drug:
            best_by_drug[drug] = col
        else:
            prev_conc = best_by_drug[drug][: best_by_drug[drug].rfind("_")].split("_")[-1]
            if conc > prev_conc:
                best_by_drug[drug] = col

    df = df[list(best_by_drug.values())]

    # ── Step 3: strip trailing time suffix ───────────────────────────────────
    df.columns = ["_".join(c.split("_")[:-1]) for c in df.columns]
    return df


# ── VIPER matrix loading ───────────────────────────────────────────────────────

DATA_DIR = "/Users/zh2477/Desktop/PANACEA_data/PANACEA_final_matrix"
OUTPUT_DIR = (
    "/Users/zh2477/Desktop/PANACEA/Mariano_Cell_line_rdata_normalized_by_ccle/"
    "normalized by ccle virper matrix considering nan/only_normalize_by_proweight_FINAL/"
)


def load_viper_matrices(data_dir):
    """
    Load all ``*vpmat.rda`` files from *data_dir* and return per-cell-line
    VIPER matrices filtered to the 24 h / highest-concentration profiles.

    Parameters
    ----------
    data_dir : str
        Directory containing ``*vpmat.rda`` files.

    Returns
    -------
    master_viper : dict
        ``{cell_line: pd.DataFrame}`` — each DataFrame includes a
        trailing ``protweight`` column.
    master_gene_list : list of str
    master_drug_list : list of str
    cell_line_list : list of str
    """
    master_gene_list = []
    master_drug_list = []
    cell_line_list = []
    master_viper = {}

    os.chdir(data_dir)

    for filepath in glob.glob("*vpmat.rda"):
        print(f"Loading {filepath} ...")
        result = pyreadr.read_r(filepath)
        df = result["vpmat"]
        protein_weights = result["protweight"]

        cell_line = filepath.partition("_")[0]
        cell_line_list.append(cell_line)

        # Drop duplicate columns (keep first occurrence)
        df = df.loc[:, ~df.columns.duplicated()]

        # Normalise underscore-heavy drug names so split("_") gives ≤ 3 parts
        new_cols = []
        for col in df.columns:
            parts = col.split("_")
            while len(parts) > 3:
                col = col.replace("_", "-", 1)
                parts = col.split("_")
            new_cols.append(col)
        df.columns = new_cols

        # Accumulate gene list
        for gene in df.index.tolist():
            if gene not in master_gene_list:
                master_gene_list.append(gene)

        # Remove control columns; accumulate drug list
        cols_to_drop = []
        for col in df.columns:
            if any(ctrl in col for ctrl in ("DMSO", "UNTREATED", "UNTREATE", "MOCK")):
                cols_to_drop.append(col)
            elif col.endswith("_6"):
                cols_to_drop.append(col)
            else:
                drug_conc = col[: col.rfind("_")]
                drug_raw = drug_conc[: drug_conc.rfind("_")]
                drug = normalize_drug_name(drug_raw)
                if drug not in master_drug_list:
                    master_drug_list.append(drug)

        df.drop(columns=cols_to_drop, inplace=True)

        # Keep closest-to-24 h, highest concentration
        df = keep_closest_24h_and_highest_conc(df)

        # Attach protein weights
        protein_weights.index = df.index
        master_viper[cell_line] = pd.concat([df, protein_weights], axis=1)

        print(f"  Done: {cell_line}")

    master_drug_list = list(set(correct_drug_names(master_drug_list)))
    print(f"\nGenes : {len(master_gene_list)}")
    print(f"Drugs : {len(master_drug_list)}")
    print(f"Cell lines: {cell_line_list}")
    return master_viper, master_gene_list, master_drug_list, cell_line_list


# ── Protein-weight re-weighting ────────────────────────────────────────────────

def reweigh_protein_activity(viper_dict, reweigh=True):
    """
    Optionally multiply each VIPER protein-activity score by its protein weight.

    Parameters
    ----------
    viper_dict : dict
        ``{cell_line: pd.DataFrame}`` where the last column is ``protweight``.
    reweigh : bool
        If ``True``, multiply activity scores by ``protweight``.

    Returns
    -------
    dict
        Same structure as *viper_dict* but without the ``protweight`` column.
    """
    result = {}
    for cell_line, df in viper_dict.items():
        df = df.copy()
        if reweigh:
            weights = df["protweight"].values[:, None]
            data = df.values * weights
            df = pd.DataFrame(data, index=df.index, columns=df.columns)
        df.drop(columns=["protweight"], inplace=True)
        result[cell_line] = df
    return result


# ── Drug-centric matrix construction ──────────────────────────────────────────

def build_drug_centric_matrices(master_viper_filtered, master_drug_list,
                                cell_line_drugs_dict):
    """
    For every drug in *master_drug_list*, concatenate its activity profile
    across all cell lines that were treated with that drug.

    Parameters
    ----------
    master_viper_filtered : dict
        ``{cell_line: pd.DataFrame}`` filtered to the common gene set.
    master_drug_list : list of str
    cell_line_drugs_dict : dict
        ``{cell_line: [drug, ...]}``

    Returns
    -------
    dict
        ``{drug: pd.DataFrame}`` where columns are cell-line names.
    """
    drug_cell_line_dict = {}

    for drug in master_drug_list:
        drug_key = f"'{drug}'"
        for cell_line, drug_list in cell_line_drugs_dict.items():
            if drug_key in drug_list:
                col = (
                    master_viper_filtered[cell_line][[drug_key]]
                    .rename(columns={drug_key: cell_line})
                )
                if drug_key not in drug_cell_line_dict:
                    drug_cell_line_dict[drug_key] = col
                else:
                    drug_cell_line_dict[drug_key] = pd.concat(
                        [drug_cell_line_dict[drug_key], col], axis=1
                    )

    return drug_cell_line_dict


# ── Hierarchical clustering & cell-line selection ─────────────────────────────

def extract_cluster_labels(dendrogram, label_key="ivl"):
    """
    Extract cluster-to-leaf-label mappings from a ``scipy`` dendrogram dict.

    Parameters
    ----------
    dendrogram : dict
        Output of ``scipy.cluster.hierarchy.dendrogram``.
    label_key : str

    Returns
    -------
    dict
        ``{color: [leaf_label, ...]}``
    """
    cluster_indices = defaultdict(list)
    for color, icoord in zip(dendrogram["color_list"], dendrogram["icoord"]):
        for leg in icoord[1:3]:
            i = (leg - 5.0) / 10.0
            if abs(i - round(i)) < 1e-5:
                cluster_indices[color].append(int(i))

    return {
        color: [dendrogram[label_key][i] for i in indices]
        for color, indices in cluster_indices.items()
    }


def cluster_and_filter_cell_lines(drug_cell_line_dict,
                                  drug_cell_line_corr_dict,
                                  plot_dir,
                                  color_threshold=1):
    """
    For each drug, hierarchically cluster its cell lines by VIPER-similarity
    and retain clusters of ≥ 2 lines (excluding the root / "C0" cluster).

    Parameters
    ----------
    drug_cell_line_dict : dict
        ``{drug: pd.DataFrame}`` from :func:`build_drug_centric_matrices`.
    drug_cell_line_corr_dict : dict
        Pre-computed pairwise VIPER-similarity matrices, keyed by drug.
    plot_dir : str
        Directory for saving cluster-map PDFs.
    color_threshold : float
        Colour-threshold passed to ``scipy.cluster.hierarchy.dendrogram``.

    Returns
    -------
    filtered : dict
        ``{drug: {cluster_id: [cell_line, ...]}}`` — drugs with ≥ 2 cell lines
        in at least one non-root cluster.
    excluded : dict
        ``{drug: [cell_line, ...]}`` — drugs that had < 2 cell lines or no
        qualifying cluster.
    """
    filtered = {}
    excluded = {}

    os.makedirs(plot_dir, exist_ok=True)

    for drug, df in drug_cell_line_dict.items():
        cell_lines = df.columns.tolist()

        if len(cell_lines) < 2:
            excluded[drug] = cell_lines
            continue

        corr_matrix = drug_cell_line_corr_dict[drug]

        # ── Cluster map ───────────────────────────────────────────────────────
        cg = sns.clustermap(
            corr_matrix,
            method="complete",
            cmap="RdBu",
            annot=True,
            annot_kws={"size": 7},
            vmin=-1,
            vmax=1,
            figsize=(15, 12),
            metric="correlation",
        )
        cg.savefig(os.path.join(plot_dir, f"{drug}.pdf"))
        plt.close("all")

        # ── Dendrogram extraction ─────────────────────────────────────────────
        plt.figure(figsize=(20, 5))
        den = scipy.cluster.hierarchy.dendrogram(
            cg.dendrogram_col.linkage,
            labels=cell_lines,
            color_threshold=color_threshold,
        )
        plt.savefig(os.path.join(plot_dir, f"{drug}_dendrogram.pdf"))
        plt.close("all")

        clusters = extract_cluster_labels(den)

        if len(cell_lines) > 2:
            for color, members in clusters.items():
                if color != "C0" and len(members) >= 2:
                    filtered.setdefault(drug, {})[color] = members
            if drug not in filtered:
                excluded[drug] = cell_lines
        else:
            # Only two cell lines — record but skip cluster assignment
            excluded[drug] = clusters.get("C0", cell_lines)

    return filtered, excluded


# ── Jaccard overlap & drug-pair scoring ───────────────────────────────────────

def jaccard(list1, list2):
    """Jaccard similarity between two lists (treated as sets)."""
    s1, s2 = set(list1), set(list2)
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    return intersection / union if union > 0 else 0.0


def find_overlapping_cluster_pairs(drug_cell_line_filtered,
                                   jaccard_threshold=1 / 3):
    """
    For every drug pair, find cluster pairs whose cell-line overlap exceeds
    *jaccard_threshold* and record the best-matching pair per drug–drug combo.

    Parameters
    ----------
    drug_cell_line_filtered : dict
        Output of :func:`cluster_and_filter_cell_lines` (``filtered`` dict).
    jaccard_threshold : float
        Minimum Jaccard score to consider two clusters as overlapping.

    Returns
    -------
    list of str
        Each entry encodes
        ``"drug1#cluster1#drug2#cluster2#jaccard_score"``.
    """
    drug_list = list(drug_cell_line_filtered.keys())
    pairs = []

    for i in range(len(drug_list)):
        for j in range(i + 1, len(drug_list)):
            d1, d2 = drug_list[i], drug_list[j]
            clusters1 = drug_cell_line_filtered[d1]
            clusters2 = drug_cell_line_filtered[d2]

            # Build Jaccard matrix (rows = clusters of d2, cols = clusters of d1)
            ref_clusters = sorted(clusters1)
            map_clusters = sorted(clusters2)

            scores = pd.DataFrame(
                {
                    ref_c: [
                        jaccard(clusters1[ref_c], clusters2[map_c])
                        for map_c in map_clusters
                    ]
                    for ref_c in ref_clusters
                },
                index=map_clusters,
            )

            # Greedily extract best non-overlapping pairs above threshold
            while scores.size > 0 and scores.max().max() >= jaccard_threshold:
                best_ref = scores.max().idxmax()
                best_map = scores[best_ref].idxmax()
                best_score = scores.loc[best_map, best_ref]

                pairs.append(
                    f"{d1}#{best_ref}#{d2}#{best_map}#{best_score:.6f}"
                )

                scores.drop(index=best_map, inplace=True)
                scores.drop(columns=best_ref, inplace=True)

    return pairs


# ── Enrichment-score aggregation ──────────────────────────────────────────────

def aggregate_enrichment_scores(cluster_pairs,
                                drug_cell_line_filtered,
                                drug_drug_similarity_dict):
    """
    For each cluster pair, average the pairwise VIPER-similarity scores over
    the overlapping cell lines (Stouffer integration).

    Parameters
    ----------
    cluster_pairs : list of str
        Output of :func:`find_overlapping_cluster_pairs`.
    drug_cell_line_filtered : dict
        ``{drug: {cluster: [cell_line, ...]}}``
    drug_drug_similarity_dict : dict
        ``{cell_line: pd.DataFrame}`` of drug-pair enrichment scores.

    Returns
    -------
    best_scores : dict
        ``{drug_pair: [cluster_label, mean_score]}``
    best_scores_stouffer : dict
        ``{drug_pair: [cluster_label, stouffer_score]}``
    """
    raw = {}
    raw_stouffer = {}

    for entry in cluster_pairs:
        parts = entry.split("#")
        d1, c1, d2, c2 = parts[0], parts[1], parts[2], parts[3]

        cell_lines_1 = drug_cell_line_filtered[d1][c1]
        cell_lines_2 = drug_cell_line_filtered[d2][c2]
        overlap = list(set(cell_lines_1) & set(cell_lines_2))

        d1_clean = d1.replace("'", "")
        d2_clean = d2.replace("'", "")
        drug_pair_key = (
            f"{d1_clean}#{d2_clean}"
            if d1_clean > d2_clean
            else f"{d2_clean}#{d1_clean}"
        )
        cluster_label = f"{c1}#{c2}"
        pair_name = (
            f"{d1}#{c1}@{d2}#{c2}"
            if d1_clean > d2_clean
            else f"{d2}#{c2}@{d1}#{c1}"
        )

        total = sum(
            float(drug_drug_similarity_dict[cl].loc[drug_pair_key]["Enrichment Score"])
            for cl in overlap
        )
        mean_score = total / len(overlap)
        stouffer_score = total / math.sqrt(len(overlap))

        raw[pair_name] = (cluster_label, mean_score)
        raw_stouffer[pair_name] = (cluster_label, stouffer_score)

    # Keep only the highest-scoring cluster assignment per drug pair
    best_scores = {}
    best_scores_stouffer = {}

    for pair_name, (cluster_label, score) in raw.items():
        d1 = pair_name.split("@")[0].split("#")[0].replace(" ", "_").replace("'", "")
        d2 = pair_name.split("@")[1].split("#")[0].replace(" ", "_").replace("'", "")
        key = f"{d1}#{d2}" if d1 > d2 else f"{d2}#{d1}"

        if key not in best_scores or best_scores[key][1] < score:
            best_scores[key] = [cluster_label, score]
            best_scores_stouffer[key] = [cluster_label, raw_stouffer[pair_name][1]]

    return best_scores, best_scores_stouffer


# ── I/O helpers ───────────────────────────────────────────────────────────────

def save_dataframe_as_rds(df, filepath):
    """
    Save a pandas DataFrame as an R ``.rds`` file via ``rpy2``.

    Parameters
    ----------
    df : pd.DataFrame
    filepath : str
    """
    from rpy2 import robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_df = robjects.conversion.py2rpy(df)

    robjects.r.assign("df_to_save", r_df)
    robjects.r(f"saveRDS(df_to_save, file='{filepath}')")


def load_viper_similarity_matrices(similarity_dir):
    """
    Read pre-computed pairwise VIPER-similarity ``.rds.rda`` files.

    Parameters
    ----------
    similarity_dir : str
        Directory containing ``*rds.rda`` files with a
        ``viper_similarity_matrix`` object.

    Returns
    -------
    dict
        ``{cell_line: pd.DataFrame}`` — symmetric similarity matrix.
    """
    similarity = {}

    for filepath in glob.glob(os.path.join(similarity_dir, "*rda")):
        print(f"Reading {filepath} ...")
        df = pyreadr.read_r(filepath)["viper_similarity_matrix"]

        # Reconstruct clean column / index names from R-mangled strings
        clean_names = []
        for name in df.columns.tolist():
            if "rds." in name:
                parts = name.split("rds.")
                clean = "-".join(parts[1].split("."))
            else:
                clean = name
            clean_names.append(clean)

        df.columns = clean_names
        df.index = clean_names

        cell_line = os.path.basename(filepath).split(" ")[0]
        similarity[cell_line] = df

    return similarity


def build_drug_pair_enrichment_dict(similarity_dict):
    """
    Convert per-cell-line similarity matrices into a flat drug-pair lookup.

    Parameters
    ----------
    similarity_dict : dict
        ``{cell_line: pd.DataFrame}``

    Returns
    -------
    dict
        ``{cell_line: pd.DataFrame}`` indexed by ``"drug1#drug2"`` strings,
        with a single column ``"Enrichment Score"``.
    """
    result = {}

    RENAME = {
        " 5z 7oxozeaenol": "(5z)7oxozeaenol",
        "BN": "BNTX maleate salt hydrate",
        "3 4methylenedioxybetanitrostyrene": "3,4methylenedioxybetanitrostyrene",
    }

    for cell_line, df in similarity_dict.items():
        matrix = (df.values + df.values.T) / 2   # symmetrise
        drugs = df.columns.tolist()

        # Apply any residual name fixes
        drugs = [RENAME.get(d, d) for d in drugs]

        pairs, scores = [], []
        for i in range(len(drugs)):
            for j in range(i + 1, len(drugs)):
                d1, d2 = drugs[i], drugs[j]
                key = f"{d1}#{d2}" if d1 > d2 else f"{d2}#{d1}"
                pairs.append(key)
                scores.append(matrix[i, j])

        result[cell_line] = pd.DataFrame(
            {"Enrichment Score": scores}, index=pairs
        )

    return result


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    # 1. Load VIPER matrices
    master_viper, master_gene_list, master_drug_list, cell_line_list = (
        load_viper_matrices(DATA_DIR)
    )

    # 2. Apply protein-weight re-weighting
    master_viper_weighted = reweigh_protein_activity(master_viper, reweigh=True)
    master_viper_unweighted = reweigh_protein_activity(master_viper, reweigh=False)

    # 3. Find common gene set; filter all matrices to it
    common_genes = None
    for df in master_viper_weighted.values():
        common_genes = (
            set(df.index) if common_genes is None else common_genes & set(df.index)
        )
    common_genes = list(common_genes)

    master_viper_filtered = {
        cl: df.loc[common_genes] for cl, df in master_viper_weighted.items()
    }

    # Per-cell-line drug lists
    cell_line_drugs = {
        cl: df.columns.tolist() for cl, df in master_viper_filtered.items()
    }

    # 4. Build drug-centric matrices and save as .rds
    drug_cell_line_matrices = build_drug_centric_matrices(
        master_viper_filtered, master_drug_list, cell_line_drugs
    )

    # 5. Load pre-computed VIPER similarities
    similarity_dir = (
        "/Users/zh2477/Desktop/drug_similarity/"
        "virper_similarity_protein_weight_normalized_FINAL/"
    )
    drug_corr_dict = load_viper_similarity_matrices(
        os.path.join(similarity_dir, "drug_centric_VIPER_matrix_FINAL",
                     "virper_distance_100genes", "virper_similarity")
    )

    # 6. Cluster cell lines per drug; filter to coherent subsets
    plot_dir = (
        "/Users/zh2477/Desktop/PANACEA_data/drug_centric_dendrogram/"
        "viper_similarity_50_complete_FINAL2/"
    )
    drug_cl_filtered, drug_cl_excluded = cluster_and_filter_cell_lines(
        drug_cell_line_matrices, drug_corr_dict, plot_dir
    )

    # 7. Find overlapping cluster pairs across drug pairs
    cluster_pairs = find_overlapping_cluster_pairs(drug_cl_filtered)

    # 8. Aggregate enrichment scores
    drug_drug_sim = build_drug_pair_enrichment_dict(
        load_viper_similarity_matrices(similarity_dir)
    )
    best_scores, best_scores_stouffer = aggregate_enrichment_scores(
        cluster_pairs, drug_cl_filtered, drug_drug_sim
    )

    print(f"\nFinal drug pairs scored     : {len(best_scores)}")
    print(f"Final drug pairs (Stouffer) : {len(best_scores_stouffer)}")

    gc.collect()
    return best_scores, best_scores_stouffer


if __name__ == "__main__":
    main()
