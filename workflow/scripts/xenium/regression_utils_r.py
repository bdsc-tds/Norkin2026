import numpy as np
import scanpy as sc
import pandas as pd
import rpy2
import scipy
from tqdm import tqdm
import warnings

# --- rpy2 Setup ---
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects import conversion

    rpy2.rinterface_lib.callbacks.logger.setLevel("ERROR")

    # Import R libraries as Python objects
    base = importr("base")
    stats = importr("stats")
    lme4 = importr("lme4")
    lmerTest = importr("lmerTest")
    MASS = importr("MASS")  # For glm.nb
    RPY2_AVAILABLE = True
except ImportError:
    warnings.warn("rpy2 or its R dependencies (lme4, lmerTest, MASS) not found. This function will not work.")
    RPY2_AVAILABLE = False

from statsmodels.stats.multitest import multipletests


def regress_gene_morphology(
    adata: sc.AnnData,
    features: list,
    fixed_effects: list = None,
    random_effects: list = None,
    family: str = "gaussian",
    layer: str = None,
) -> pd.DataFrame:
    """
    Performs regression using R's lme4 (for mixed models) or stats/MASS (for fixed-effects models).

    Args:
        adata (anndata.AnnData): AnnData object with expression data.
        features (list): List of columns in .obs to test as predictors.
        fixed_effects (list, optional): Covariates in .obs to use as fixed effects.
        random_effects (list, optional): Grouping factors for random effects. If None, a standard
                                      GLM/LM is fitted. Defaults to None.
        family (str, optional): 'gaussian' (log-data) or 'negativebinomial' (raw counts).
        layer (str, optional): Layer in adata to use. If None, uses adata.X.

    Returns:
        pd.DataFrame: A tidy DataFrame with the full R model summary for fixed effects.
    """
    if not RPY2_AVAILABLE:
        raise ImportError("Please install rpy2 and R dependencies to use this function.")

    # 1. --- Prepare a unified DataFrame for modeling ---
    model_type = "Mixed-Effects Model" if random_effects else "Fixed-Effects Model (GLM/LM)"
    print(f"--- Running R {model_type} via rpy2 (family: {family}) ---")

    all_effects = (features or []) + (fixed_effects or []) + (random_effects or [])
    model_data = adata.obs[list(set(all_effects))].copy()
    for col in (fixed_effects or []) + (random_effects or []):
        model_data[col] = model_data[col].astype("category")

    expression_data = adata.layers[layer] if layer else adata.X

    # 2. --- Main Loop ---
    all_results_list = []
    failure_count = 0

    for feature in tqdm(features, desc="Testing morphology features"):
        # Build the R formula dynamically
        formula_parts = [feature] + (fixed_effects or [])
        if random_effects:
            formula_parts.extend([f"(1 | {eff})" for eff in random_effects])
        formula_str = f"gene_expression ~ {' + '.join(formula_parts)}"
        formula_r = ro.Formula(formula_str)

        print(f"Using formula: {formula_str}")

        for gene in tqdm(adata.var_names, desc=f"Fitting for {feature}"):
            gene_model_df = model_data.copy()
            gene_model_df["gene_expression"] = expression_data[:, adata.var_names == gene].toarray().flatten()

            try:
                with conversion.localconverter(ro.default_converter + pandas2ri.converter):
                    model_data_r = ro.conversion.py2rpy(gene_model_df)

                # --- INTELLIGENT MODEL SELECTION ---
                if random_effects:
                    # Mixed-Effects Path
                    if family == "gaussian":
                        model_fit = lmerTest.lmer(formula_r, data=model_data_r)
                    elif family == "negativebinomial":
                        model_fit = lme4.glmer_nb(formula_r, data=model_data_r)
                else:
                    # Fixed-Effects Only Path (GLM / LM)
                    if family == "gaussian":
                        model_fit = stats.lm(formula_r, data=model_data_r)
                    elif family == "negativebinomial":
                        model_fit = MASS.glm_nb(formula_r, data=model_data_r)
                # --- END MODEL SELECTION ---

                summary_obj = base.summary(model_fit)
                coef_matrix = summary_obj.rx2("coefficients")

                summary_df = pd.DataFrame(
                    np.array(coef_matrix),
                    index=list(ro.r["rownames"](coef_matrix)),
                    columns=list(ro.r["colnames"](coef_matrix)),
                )

                summary_df["gene"] = gene
                summary_df["feature_tested"] = feature
                all_results_list.append(summary_df.reset_index().rename(columns={"index": "term"}))

            except Exception:
                failure_count += 1

    # 3. --- Compile and Finalize Results ---
    if failure_count > 0:
        print(f"Total model failures: {failure_count}")
    if not all_results_list:
        print("Warning: All models failed.")
        return pd.DataFrame()

    results_df = pd.concat(all_results_list, ignore_index=True)

    pval_col = next((col for col in results_df.columns if "Pr" in col), None)
    if pval_col:
        results_df.rename(columns={pval_col: "p_val"}, inplace=True)
        q_vals = results_df.groupby("feature_tested")["p_val"].transform(
            lambda p: multipletests(p, method="fdr_bh")[1] if p.notna().any() else p
        )
        results_df["q_val"] = q_vals
        return results_df.sort_values(by="p_val")
    else:
        return results_df


def _create_synthetic_anndata_for_testing(n_cells=2000, n_genes=100, n_organoids=20, n_samples=4):
    """Creates a synthetic AnnData object with a known hierarchical structure and ground truth."""
    print("--- Creating synthetic dataset for validation ---")
    adata = sc.datasets.blobs(n_observations=n_cells, n_variables=n_genes)
    adata.X = np.abs(adata.X * 10).astype(int)  # Make into integer counts

    # Create hierarchical metadata: cells -> organoids -> samples
    adata.obs["organoid_id"] = [f"Org_{i}" for i in np.random.randint(0, n_organoids, n_cells)]
    org_to_sample = {f"Org_{i}": f"Sample_{i % n_samples}" for i in range(n_organoids)}
    adata.obs["sample_batch"] = adata.obs["organoid_id"].map(org_to_sample)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    # Create organoid-level morphology
    organoid_meta = pd.DataFrame(
        {"area": np.random.uniform(0.1, 1, n_organoids), "circularity": np.random.uniform(0.1, 1, n_organoids)},
        index=[f"Org_{i}" for i in range(n_organoids)],
    )
    adata.obs = adata.obs.assign(**organoid_meta.loc[adata.obs["organoid_id"]].set_index(adata.obs.index))

    # Inject ground truth signals on top of baseline counts
    ground_truth = {
        ("gene_10", "area"): {"coeff": 50.0, "desc": "Strong positive"},
        ("gene_25", "circularity"): {"coeff": -40.0, "desc": "Strong negative"},
    }
    print("Injecting ground truth signals:")
    for (gene, morph), params in ground_truth.items():
        gene_idx = adata.var_names.get_loc(gene)
        signal = (adata.obs[morph] * params["coeff"]).astype(int)
        adata.X[:, gene_idx] = np.maximum(0, adata.X[:, gene_idx] + signal)
        print(f"  - Associating '{gene}' with '{morph}'")

    adata.X = scipy.sparse.csr_matrix(adata.X)
    return adata, ground_truth


def _validate_results(results_df, ground_truth):
    """Checks if the results dataframe correctly identified the ground truth."""
    if results_df.empty:
        print("  ❌ FAILURE: Results DataFrame is empty. All models may have failed.")
        return False

    top_hits = results_df.head(10).set_index(["gene", "feature_tested"])
    all_ok = True
    for (gene, morph), params in ground_truth.items():
        print(f"Checking for: {gene} ~ {morph} ({params['desc']})")
        if (gene, morph) in top_hits.index:
            hit = top_hits.loc[(gene, morph)]

            # Validation logic depends on the model type
            if "q_val" in hit.index:  # Statsmodels output
                is_significant = hit["q_val"] < 0.01
                sign_ok = np.sign(hit["coefficient"]) == np.sign(params["coeff"])
                metric = f"q-val: {hit['q_val']:.2e}"

            if is_significant and sign_ok:
                print(f"  ✅ SUCCESS: Found in top hits with correct sign ({metric})")
            else:
                print(f"  ❌ FAILURE: Found but significance/sign was incorrect ({metric})")
                all_ok = False
        else:
            print("  ❌ FAILURE: Ground truth pair not found in top 10 significant results.")
            all_ok = False
    return all_ok


# --- Run the validation suite ---
if __name__ == "__main__":
    adata_synth, ground_truth = _create_synthetic_anndata_for_testing()

    # Best practice: Filter lowly expressed genes before running models
    sc.pp.filter_genes(adata_synth, min_cells=10)
    sc.pp.normalize_total(adata_synth, target_sum=1e4)
    sc.pp.log1p(adata_synth)

    features_to_test = ["area", "circularity"]

    print("\n--- Running LMM on log-transformed data ---")
    results = regress_gene_morphology(
        adata=adata_synth,
        features=features_to_test,
        family="gaussian",
    )

    _validate_results(
        results,
        ground_truth,
    )
