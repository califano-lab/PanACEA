"""
PANACEA Drug Analysis Pipeline
================================
Loads VIPER protein activity matrices, normalizes drug names,
performs hierarchical clustering across cell lines, and computes
drug-drug similarity scores via Stouffer and mean aggregation.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import gc
import glob
import math
import os
import copy
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy
import seaborn as sns
import pyreadr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import gseapy as gp
from gseapy.parser import Biomart

# ── Inline backend (Jupyter) ──────────────────────────────────────────────────
# %matplotlib inline
# %config InlineBackend.figure_format='retina'
# %load_ext autoreload
# %autoreload 2


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

# Maps non-canonical drug names → canonical names.
DRUG_ALIAS: Dict[str, str] = {
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

JACCARD_THRESHOLD = 1 / 3
DENDROGRAM_COLOR_THRESHOLD = 1
CLUSTERMAP_FIG_SIZE = (15, 12)
DENDROGRAM_FIG_SIZE = (20, 5)


# ══════════════════════════════════════════════════════════════════════════════
# Drug-name utilities
# ══════════════════════════════════════════════════════════════════════════════

def normalize_drug_name(raw: str) -> str:
    """Return a cleaned, capitalised drug name with hyphens and slashes removed."""
    name = raw.replace("/", "-").replace("-", "")
    return name.capitalize()


def canonicalize_drug_names(names: List[str]) -> List[str]:
    """Replace known aliases with their canonical names (in-place copy)."""
    result = []
    for name in names:
        canonical = DRUG_ALIAS.get(name, name)
        if canonical != name:
            print(f"  {name!r} → {canonical!r}")
        result.append(canonical)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# DataFrame helpers
# ══════════════════════════════════════════════════════════════════════════════

def _duplicate_indices(items: list) -> Tuple[dict, list]:
    """
    Return (dict of item → [indices], flat list of all duplicated indices).
    Only items that appear more than once are included.
    """
    seen: Dict[str, List[int]] = {}
    for idx, item in enumerate(items):
        seen.setdefault(item, []).append(idx)

    duplicates = {k: v for k, v in seen.items() if len(v) > 1}
    flat = [i for indices in duplicates.values() for i in indices]
    return duplicates, flat


def remove_concentration_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average columns that share the same drug+concentration label
    (i.e. differ only in the trailing time-point token).
    """
    cols = df.columns.tolist()
    base_names = [c.rsplit("_", 1)[0] for c in cols]

    duplicates, dup_indices = _duplicate_indices(base_names)

    # Rename non-duplicate columns to their clean base name
    new_cols = [
        base_names[i] if i not in dup_indices else cols[i]
        for i in range(len(cols))
    ]
    df.columns = new_cols

    # Average duplicated columns, then drop originals
    for base, indices in duplicates.items():
        avg = sum(df[cols[i]] for i in indices) / len(indices)
        for i in indices:
            del df[cols[i]]
        df[base] = avg

    return df


def remove_non24hr_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each drug+concentration, keep the time-point closest to 24 h.
    Among remaining duplicates for the same drug, keep the highest concentration.
    Finally, strip the time-point suffix from column names.
    """
    cols = df.columns.tolist()

    # ── Step 1: keep the time point closest to 24 h ──────────────────────────
    best_by_drug_conc = {}
    for col in cols:

        drug_conc = col[: col.rfind("_")]

        if drug_conc not in best_by_drug_conc:
            best_by_drug_conc[drug_conc] = col
        else:
            prev_time = float(best_by_drug_conc[drug_conc].split("_")[-1])
            time = float(col.split("_")[-1])
            if abs(time - 24) < abs(prev_time - 24):
                best_by_drug_conc[drug_conc] = col

    df = df[list(best_by_drug_conc.values())]

    # ── Step 2: pick highest concentration ───────────────────────────────────
    best_by_drug: Dict[str, str] = {}
    for col in df.columns:
        base_no_time = col.rsplit("_", 1)[0]   # drug_conc
        drug = "_".join(base_no_time.split("_")[:-1])
        conc = base_no_time.split("_")[-1]
        prev = best_by_drug.get(drug)
        if prev is None or conc > prev.rsplit("_", 2)[-2]:
            best_by_drug[drug] = col

    df = df[list(best_by_drug.values())]

    # ── Step 3: strip time suffix ─────────────────────────────────────────────
    df.columns = ["_".join(c.split("_")[:-1]) for c in df.columns]
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Gene-ID conversion
# ══════════════════════════════════════════════════════════════════════════════

def entrez_to_symbol(entrez_ids: List[str], max_retries: int = 5, backoff: int = 3) -> Dict[str, str]:
    """
    Map Entrez IDs → HGNC gene symbols.
    Tries Biomart first (with exponential backoff), then falls back to mygene.
    """
    last_err = None

    for attempt in range(max_retries):
        try:
            bm = Biomart()
            res = bm.query(
                dataset="hsapiens_gene_ensembl",
                attributes=["external_gene_name", "entrezgene_id"],
                filters={"entrezgene_id": entrez_ids},
            )
            if not {"entrezgene_id", "external_gene_name"}.issubset(res.columns):
                raise ValueError(f"Unexpected Biomart columns: {res.columns.tolist()}")

            mapping = {}
            for eid, sym in zip(res["entrezgene_id"], res["external_gene_name"]):
                if pd.notna(eid):
                    mapping.setdefault(str(int(eid)), sym)
            print(f"Biomart succeeded (attempt {attempt + 1})")
            return mapping

        except Exception as exc:
            last_err = exc
            wait = backoff * (2 ** attempt)
            print(f"Biomart attempt {attempt + 1}/{max_retries} failed: {exc}. Retrying in {wait}s…")
            time.sleep(wait)

    # Fallback: mygene
    print(f"Biomart exhausted. Falling back to mygene…")
    try:
        import mygene
        mg = mygene.MyGeneInfo()
        hits = mg.querymany(entrez_ids, scopes="entrezgene", fields="symbol", species="human", silent=True)
        mapping = {str(h["query"]): h["symbol"] for h in hits if "symbol" in h}
        print("mygene fallback succeeded.")
        return mapping
    except Exception as exc2:
        raise RuntimeError(
            f"Both Biomart and mygene failed.\n"
            f"  Biomart: {last_err}\n  mygene: {exc2}"
        ) from exc2


def relabel_index_entrez_to_symbol(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with its index converted from Entrez IDs to gene symbols."""
    mapping = entrez_to_symbol(df.index.tolist())
    new_index = [mapping.get(i, i) for i in df.index]
    return pd.DataFrame(df.values, index=new_index, columns=df.columns)


# ══════════════════════════════════════════════════════════════════════════════
# Protein-activity weighting
# ══════════════════════════════════════════════════════════════════════════════

def apply_protein_weights(
    viper_dict: Dict[str, pd.DataFrame],
    reweigh: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Multiply each cell line's activity matrix by its protein-weight column,
    then drop that column. If *reweigh* is False, only the column is dropped.
    """
    result = {}
    for cell_line, df in viper_dict.items():
        mat = df.values
        if reweigh:
            weights = mat[:, -1, np.newaxis]
            mat = mat * weights
        new_df = pd.DataFrame(mat, index=df.index, columns=df.columns)
        new_df.drop(columns=["protweight"], inplace=True)
        result[cell_line] = new_df
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Drug-name correction on DataFrames
# ══════════════════════════════════════════════════════════════════════════════

def _col_to_canonical(col: str) -> str:
    """Convert a raw 'drug_conc_time' column label to a quoted canonical name."""
    base = "_".join(col.split("_")[:-1])          # strip time
    name = normalize_drug_name(base)
    name = DRUG_ALIAS.get(name, name)
    return f"'{name}'"


def correct_drug_names_in_dict(viper_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Strip trailing concentration token from column names, apply canonical
    drug names, and remove CDDO-Me columns (duplicate of Bardoxolone methyl).
    """
    result = {}
    for cell_line, df in viper_dict.items():
        df = df.filter(regex="^(?!.*CDDO-Me)")   # drop CDDO-Me duplicates
        df = df.copy()
        df.columns = [_col_to_canonical(c) for c in df.columns]
        result[cell_line] = df
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Similarity & clustering utilities
# ══════════════════════════════════════════════════════════════════════════════

def jaccard(a: list, b: list) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def get_cluster_classes(den: dict, label: str = "ivl") -> Dict[str, list]:
    """Extract cluster → member labels from a scipy dendrogram dict."""
    cluster_idxs: Dict[str, List[int]] = defaultdict(list)
    for color, icoord in zip(den["color_list"], den["icoord"]):
        for leg in icoord[1:3]:
            i = (leg - 5.0) / 10.0
            if abs(i - round(i)) < 1e-5:
                cluster_idxs[color].append(int(i))

    return {c: [den[label][i] for i in idxs] for c, idxs in cluster_idxs.items()}

def _fix_drug_col_underscores(cols: List[str]) -> List[str]:
    """
    Some drug names contain underscores; normalise so every column has exactly
    3 tokens (drug_conc_time) by replacing extra underscores with hyphens.
    """
    fixed = []
    for col in cols:
        parts = col.split("_")
        while len(parts) > 3:
            col = col.replace("_", "-", 1)
            parts = col.split("_")
        fixed.append(col)
    return fixed


def load_viper_matrices(data_dir: str) -> Tuple[Dict, List, List, List]:
    """
    Load all *_vpmat.rda files from *data_dir*.

    Returns
    -------
    viper_dict      : cell_line → DataFrame (activity × drug, with protweight col)
    master_gene_list
    master_drug_list
    cell_line_list
    """
    os.chdir(data_dir)

    master_gene_list: List[str] = []
    master_drug_list: List[str] = []
    cell_line_list: List[str] = []
    viper_dict: Dict[str, pd.DataFrame] = {}

    for path in glob.glob("*vpmat.rda"):
        print(f"Loading {path}…")
        result = pyreadr.read_r(path)
        df = result["vpmat"]
        protein_weight = result["protweight"]
        cell_line = path.split("_")[0]
        cell_line_list.append(cell_line)

        # De-duplicate identical columns (keep first)
        df = df.loc[:, ~df.columns.duplicated()]

        # Normalise column underscores
        df.columns = _fix_drug_col_underscores(df.columns.tolist())

        gene_list  = df.index.tolist()
        drug_cols  = df.columns.tolist()

        # Update master gene list
        master_gene_list = list(set(master_gene_list) | set(gene_list))

        # Filter columns and build master drug list
        to_drop = []
        for col in drug_cols:
            if any(kw in col for kw in ("DMSO", "UNTREATED", "UNTREATE", "MOCK")):
                to_drop.append(col)
            elif col.endswith("_6"):
                to_drop.append(col)
            else:
                drug_base = "_".join(col.split("_")[:-2])   # strip conc + time
                clean = normalize_drug_name(drug_base)
                if clean not in master_drug_list:
                    master_drug_list.append(clean)

        df.drop(columns=to_drop, inplace=True, errors="ignore")
        df = remove_non24hr_time(df)

        # Attach protein weights
        protein_weight.index = gene_list
        df = pd.concat([df, protein_weight], axis=1)

        viper_dict[cell_line] = df
        print(f"  {path} done.")

    master_drug_list = list(set(canonicalize_drug_names(master_drug_list)))
    return viper_dict, master_gene_list, master_drug_list, cell_line_list


def load_viper_similarities(sim_dir: str) -> Tuple[Dict, Dict]:
    """
    Read pre-computed per-drug VIPER similarity matrices.

    Returns
    -------
    corr_dict     : drug → correlation DataFrame (cell_line × cell_line)
    distance_dict : drug → distance DataFrame (1 − correlation)
    """
    corr_dict:     Dict[str, pd.DataFrame] = {}
    distance_dict: Dict[str, pd.DataFrame] = {}

    for path in glob.glob(os.path.join(sim_dir, "*.rds.rda")):
        result = pyreadr.read_r(path)
        sim_df = result["viper_similarity_matrix_in_correlation"]

        # Clean cell-line labels produced by R
        cleaned = []
        for name in sim_df.columns:
            if "rds" in name:
                real = "-".join(name.split(".rds.")[1].split("."))
                cleaned.append(real)
            else:
                cleaned.append(name)
        sim_df.columns = sim_df.index = cleaned

        drug = os.path.basename(path).split(".rds.")[0]
        if drug in corr_dict:
            print(f"Warning: duplicate drug key {drug!r}")
        corr_dict[drug]     = sim_df
        distance_dict[drug] = 1 - sim_df

    return corr_dict, distance_dict


# ══════════════════════════════════════════════════════════════════════════════
# Hierarchical clustering & cell-line selection
# ══════════════════════════════════════════════════════════════════════════════

def cluster_cell_lines_per_drug(
    drug_cell_line_df_dict: Dict[str, pd.DataFrame],
    corr_dict: Dict[str, pd.DataFrame],
    out_dir: str,
) -> Tuple[Dict, Dict]:
    """
    For each drug, perform hierarchical clustering over cell lines using the
    pre-computed VIPER similarity matrix.  Clusters with ≥ 2 members are kept.

    Returns
    -------
    kept    : drug → {cluster_id: [cell_lines]}
    filtered_out : drug → [cell_lines]  (drugs with < 2 cell lines or no cluster)
    """
    kept: Dict[str, dict] = {}
    filtered_out: Dict[str, list] = {}

    for drug, df in drug_cell_line_df_dict.items():
        cell_lines = df.columns.tolist()

        if len(cell_lines) < 2:
            filtered_out[drug] = cell_lines
            continue

        # ── Clustermap ───────────────────────────────────────────────────────
        g = sns.clustermap(
            corr_dict[drug],
            method="complete", cmap="RdBu",
            annot=True, annot_kws={"size": 7},
            vmin=-1, vmax=1, figsize=CLUSTERMAP_FIG_SIZE,
            metric="correlation",
        )
        g.savefig(os.path.join(out_dir, f"{drug}.pdf"))
        plt.close("all")

        # ── Dendrogram ───────────────────────────────────────────────────────
        plt.figure(figsize=DENDROGRAM_FIG_SIZE)
        den = scipy.cluster.hierarchy.dendrogram(
            g.dendrogram_col.linkage,
            labels=cell_lines,
            color_threshold=DENDROGRAM_COLOR_THRESHOLD,
        )
        plt.savefig(os.path.join(out_dir, f"{drug}_dendrogram.pdf"))
        plt.close("all")

        clusters = get_cluster_classes(den)

        if len(cell_lines) == 2:
            filtered_out[drug] = clusters.get("C0", cell_lines)
            continue

        # Keep non-noise clusters (not "C0") with at least 2 members
        drug_clusters = {
            cid: members
            for cid, members in clusters.items()
            if cid != "C0" and len(members) >= 2
        }
        if drug_clusters:
            kept[drug] = drug_clusters
        else:
            filtered_out[drug] = cell_lines

    return kept, filtered_out


# ══════════════════════════════════════════════════════════════════════════════
# Drug-drug Jaccard mapping
# ══════════════════════════════════════════════════════════════════════════════

def map_drug_clusters_by_jaccard(
    drug_cluster_dict: Dict[str, dict],
) -> List[str]:
    """
    For every drug-pair, find cluster pairs with Jaccard ≥ JACCARD_THRESHOLD.
    Returns a list of strings: 'drug1#cluster1#drug2#cluster2#score'.
    """
    drugs = list(drug_cluster_dict.keys())
    records: List[str] = []

    for i, drug1 in enumerate(drugs):
        for drug2 in drugs[i + 1:]:
            clusters1 = drug_cluster_dict[drug1]
            clusters2 = drug_cluster_dict[drug2]

            # Always iterate over the smaller cluster set as reference
            if len(clusters1) > len(clusters2):
                clusters1, clusters2 = clusters2, clusters1
                drug1, drug2 = drug2, drug1

            # Build a Jaccard matrix (rows = clusters2, cols = clusters1)
            col_ids = list(clusters1.keys())
            row_ids = list(clusters2.keys())
            matrix = pd.DataFrame(
                [[jaccard(clusters1[c1], clusters2[c2]) for c1 in col_ids] for c2 in row_ids],
                index=row_ids, columns=col_ids,
            )

            # Greedy matching above threshold
            while matrix.size and matrix.max().max() >= JACCARD_THRESHOLD:
                best_col = matrix.max().idxmax()
                best_row = matrix[best_col].idxmax()
                score    = matrix.loc[best_row, best_col]
                records.append(f"{drug1}#{best_col}#{drug2}#{best_row}#{score}")
                matrix.drop(columns=[best_col], inplace=True)
                matrix.drop(index=[best_row],   inplace=True)

    return records


# ══════════════════════════════════════════════════════════════════════════════
# VIPER-similarity aggregation
# ══════════════════════════════════════════════════════════════════════════════

def _clean_drug_name_for_lookup(drug_col: str) -> str:
    """Reverse the R-induced name mangling in viper-similarity file columns."""
    name = drug_col.replace(" 5z 7oxozeaenol", "(5z)7oxozeaenol")
    name = name.replace("BN", "BNTX maleate salt hydrate")
    name = name.replace("3 4methylenedioxybetanitrostyrene", "3,4methylenedioxybetanitrostyrene")
    return name


def load_weighted_viper_similarities(sim_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load protein-weight-normalised VIPER similarity matrices (one per cell line).

    Returns
    -------
    dict : cell_line → DataFrame indexed by drug-pair strings, column 'Enrichment Score'
    """
    os.chdir(sim_dir)
    result_dict: Dict[str, pd.DataFrame] = {}

    for path in glob.glob("*rda"):
        print(f"Processing {path}…")
        df = pyreadr.read_r(path)["viper_similarity_matrix"]
        cell_line = path.split(" ")[0]

        # Decode column names (R → Python)
        raw_drugs = df.columns.tolist()
        drugs = []
        for d in raw_drugs:
            actual = d.split("rds.")[1].split("X.")[1][:-1].replace(".", " ")
            drugs.append(actual)
        df.columns = df.index = drugs

        # Symmetrise (average with transpose) and build long-form index
        mat = (df.values + df.values.T) / 2
        pairs, scores = [], []
        for i in range(len(drugs)):
            for j in range(i + 1, len(drugs)):
                d1 = _clean_drug_name_for_lookup(drugs[i])
                d2 = _clean_drug_name_for_lookup(drugs[j])
                pair = (d1 + "#" + d2) if d1 > d2 else (d2 + "#" + d1)
                pairs.append(pair)
                scores.append(mat[i, j])

        result_dict[cell_line] = pd.DataFrame({"Enrichment Score": scores}, index=pairs)

    return result_dict


def aggregate_drug_pair_scores(
    cluster_records: List[str],
    drug_cluster_dict: Dict[str, dict],
    similarity_dict: Dict[str, pd.DataFrame],
) -> Tuple[Dict, Dict]:
    """
    For each cluster-pair record, average VIPER similarity over the overlapping
    cell lines.  Returns two dicts keyed by canonical drug-pair strings:
      - mean_dict   : value = mean enrichment score
      - stouffer_dict: value = Stouffer-combined score (sum / sqrt(n))
    """
    raw_mean:     Dict[str, tuple] = {}   # drug_pair → (cluster_tag, score)
    raw_stouffer: Dict[str, tuple] = {}

    for record in cluster_records:
        d1, c1, d2, c2, _ = record.split("#")
        cells1 = drug_cluster_dict[d1][c1]
        cells2 = drug_cluster_dict[d2][c2]
        overlap = list(set(cells1) & set(cells2))

        d1_clean = d1.strip("'")
        d2_clean = d2.strip("'")
        lookup_pair = (d1_clean + "#" + d2_clean) if d1_clean > d2_clean else (d2_clean + "#" + d1_clean)
        cluster_tag = f"{c1}#{c2}"

        total = sum(
            float(similarity_dict[cl].loc[lookup_pair, "Enrichment Score"])
            for cl in overlap
        )
        mean_score     = total / len(overlap)
        stouffer_score = total / math.sqrt(len(overlap))

        # Canonical drug-pair key
        dk1 = d1_clean.replace(" ", "_")
        dk2 = d2_clean.replace(" ", "_")
        key = (dk1 + "#" + dk2) if dk1 > dk2 else (dk2 + "#" + dk1)

        for store, score in [(raw_mean, mean_score), (raw_stouffer, stouffer_score)]:
            if key not in store or store[key][1] < score:
                store[key] = (cluster_tag, score)

    mean_dict     = {k: v for k, v in raw_mean.items()}
    stouffer_dict = {k: v for k, v in raw_stouffer.items()}
    return mean_dict, stouffer_dict


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ══════════════════════════════════════════════════════════════════════════════
    # Data loading
    # ══════════════════════════════════════════════════════════════════════════════

    VIPER_DATA_DIR = "data/PANACEA_final_matrix"
    VIPER_SIM_DIR = "data/cell_line_similarity"
    WEIGHTED_SIM_DIR = "data/drug_similarity"
    DENDROGRAM_OUT_DIR = "data/output/drug_centric_dendrogram/"
    OUTPUT_DIR_1 = "data/output"


    # ── 1. Load raw VIPER matrices ────────────────────────────────────────────
    raw_viper, master_genes, master_drugs, cell_lines = load_viper_matrices(VIPER_DATA_DIR)
    print(f"Genes: {len(master_genes)}  |  Drugs: {len(master_drugs)}  |  Cell lines: {len(cell_lines)}")

    # ── 2. Weight (or not) protein activity ───────────────────────────────────
    weighted     = apply_protein_weights(raw_viper, reweigh=True)
    not_weighted = apply_protein_weights(raw_viper, reweigh=False)

    # ── 3. Correct drug column names ──────────────────────────────────────────
    weighted_clean     = correct_drug_names_in_dict(weighted)
    not_weighted_clean = correct_drug_names_in_dict(not_weighted)

    # ── 4. Find common gene set; filter to it ─────────────────────────────────
    common_genes: Set[str] = set.intersection(*(set(df.index) for df in weighted_clean.values()))
    filtered: Dict[str, pd.DataFrame] = {
        cl: df.loc[list(common_genes)] for cl, df in weighted_clean.items()
    }

    # ── 5. Build drug → cell_line DataFrame dict ───────────────────────────────
    cell_line_drugs: Dict[str, list] = {cl: df.columns.tolist() for cl, df in filtered.items()}

    drug_cl_dict: Dict[str, pd.DataFrame] = {}
    for drug in [f"'{d}'" for d in master_drugs]:
        for cl, drug_list in cell_line_drugs.items():
            if drug in drug_list:
                col = filtered[cl][[drug]].rename(columns={drug: cl})
                if drug not in drug_cl_dict:
                    drug_cl_dict[drug] = col
                else:
                    drug_cl_dict[drug] = pd.concat([drug_cl_dict[drug], col], axis=1)

    # ── 6. Load pre-computed VIPER similarities ────────────────────────────────
    corr_dict, _ = load_viper_similarities(VIPER_SIM_DIR)

    # ── 7. Cluster cell lines per drug ────────────────────────────────────────
    kept_clusters, _ = cluster_cell_lines_per_drug(drug_cl_dict, corr_dict, DENDROGRAM_OUT_DIR)
    print(f"Drugs with valid clusters: {len(kept_clusters)}")

    # ── 8. Jaccard mapping across drug pairs ──────────────────────────────────
    cluster_records = map_drug_clusters_by_jaccard(kept_clusters)
    print(f"Cluster-pair records: {len(cluster_records)}")

    # ── 9. Load weighted similarities & aggregate scores ──────────────────────
    weighted_sim = load_weighted_viper_similarities(WEIGHTED_SIM_DIR)
    mean_scores, stouffer_scores = aggregate_drug_pair_scores(
        cluster_records, kept_clusters, weighted_sim
    )
    print(f"Drug pairs (mean):     {len(mean_scores)}")
    print(f"Drug pairs (Stouffer): {len(stouffer_scores)}")

    gc.collect()


if __name__ == "__main__":
    main()