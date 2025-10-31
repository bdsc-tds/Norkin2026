import numpy as np
import scanpy as sc
import scipy
import pandas as pd
import warnings
from tqdm import tqdm
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests

# rpy2 Setup
try:
    import rpy2
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects import conversion, r

    rpy2.rinterface_lib.callbacks.logger.setLevel("ERROR")

    # Import R libraries
    base = importr("base")
    stats = importr("stats")
    lme4 = importr("lme4")
    lmerTest = importr("lmerTest")
    MASS = importr("MASS")
    RPY2_AVAILABLE = True
except ImportError:
    warnings.warn("rpy2 or its R dependencies (lme4, lmerTest, MASS) not found.")
    RPY2_AVAILABLE = False


def _fit_model_for_gene(
    gene_name: str,
    gene_expression: np.ndarray,
    model_data: pd.DataFrame,
    formula_r: ro.Formula,
    family: str,
    random_effects: list,
    feature: str,
) -> pd.DataFrame or None:
    """Helper function to fit a model for a single gene. For use with joblib."""

    gene_model_df = model_data.copy()
    gene_model_df["gene_expression"] = gene_expression

    try:
        with conversion.localconverter(ro.default_converter + pandas2ri.converter):
            model_data_r = ro.conversion.py2rpy(gene_model_df)

        # MODEL SELECTION
        if random_effects:
            model_fit = (
                lmerTest.lmer(formula_r, data=model_data_r)
                if family == "gaussian"
                else lme4.glmer_nb(formula_r, data=model_data_r)
            )
        else:
            model_fit = (
                stats.lm(formula_r, data=model_data_r)
                if family == "gaussian"
                else MASS.glm_nb(formula_r, data=model_data_r)
            )

        # Extract results
        summary_obj = base.summary(model_fit)
        coef_matrix = summary_obj.rx2("coefficients")

        summary_df = pd.DataFrame(
            np.array(coef_matrix),
            index=list(r["rownames"](coef_matrix)),
            columns=list(r["colnames"](coef_matrix)),
        )

        summary_df["gene"] = gene_name
        summary_df["feature_tested"] = feature
        summary_df = summary_df.reset_index().rename(columns={"index": "term"})

        return summary_df

    except Exception as e:
        # warnings.warn(f"Model failed for gene {gene_name} with feature {feature}: {e}")
        return None


def regress_gene_morphology(
    adata: sc.AnnData,
    morph_features: list,
    fixed_effects: list = None,
    random_effects: list = None,
    family: str = "gaussian",
    layer: str = None,
    n_jobs: int = -1,
    return_r: bool = False,
) -> pd.DataFrame:
    """
    Performs regression in parallel for each gene using R's lme4 or stats/MASS.

    Args:
        adata (sc.AnnData): AnnData object with expression data.
        morph_features (list): List of columns in .obs to test as primary predictors.
        fixed_effects (list, optional): Covariates in .obs to use as fixed effects.
        random_effects (list, optional): Grouping factors for random effects. If None,
                                      a standard GLM/LM is fitted. Defaults to None.
        family (str, optional): 'gaussian' (for log-normalized data) or
                                'negativebinomial' (for raw counts). Defaults to 'gaussian'.
        layer (str, optional): Layer in adata to use for expression. If None, uses adata.X.
        n_jobs (int, optional): Number of CPU cores to use for parallel processing.
                                -1 means using all available cores. Defaults to -1.

    Returns:
        pd.DataFrame: A tidy DataFrame with model results for all genes and morph_features.
    """
    if not RPY2_AVAILABLE:
        raise ImportError("Please install rpy2 and R dependencies to use this function.")
    morph_features = list(morph_features)

    # 1. Prepare data
    model_type = "Mixed-Effects Model" if random_effects else "Fixed-Effects Model (GLM/LM)"
    print(f"Running R {model_type} via rpy2 (family: {family}) in parallel")

    all_effects = list(set((morph_features or []) + (fixed_effects or []) + (random_effects or [])))
    model_data = adata.obs[all_effects].copy()

    expression_data = adata.layers[layer] if layer is not None else adata.X
    if scipy.sparse.issparse(expression_data):
        expression_data = expression_data.toarray()

    # 2. Loop over morph_features, parallelize over genes
    all_results_list = []
    total_failures = 0

    for feature in tqdm(morph_features, desc="Testing morphology morph_features"):
        # Build the R formula for the current feature
        formula_parts = [feature] + (fixed_effects or [])
        if random_effects:
            formula_parts.extend([f"(1 | {eff})" for eff in random_effects])

        formula_str = f"gene_expression ~ {' + '.join(formula_parts)}"
        formula_r = ro.Formula(formula_str)
        print(f"Using formula: {formula_str}")

        # Run model fitting in parallel for all genes for the current feature
        results = Parallel(n_jobs=n_jobs)(
            delayed(_fit_model_for_gene)(
                adata.var_names[i],
                expression_data[:, i],
                model_data,
                formula_r,
                family,
                random_effects,
                feature,
            )
            for i in tqdm(range(adata.n_vars), desc=f"Fitting models for {feature}", leave=False)
        )

        # Filter out failed models (which return None)
        successful_results = [res for res in results if res is not None]
        total_failures += len(results) - len(successful_results)
        all_results_list.extend(successful_results)

    # 3. Compile and finalize results
    if total_failures > 0:
        print(f"\nTotal model failures across all morph_features: {total_failures}")

    results_df = pd.concat(all_results_list, ignore_index=True)

    # Calculate q-values (FDR)
    pval_col = next((col for col in results_df.columns if "Pr" in col), None)
    if pval_col:
        results_df.rename(columns={pval_col: "p_val"}, inplace=True)
        # Calculate q-values per feature tested
        q_vals = results_df.groupby("feature_tested")["p_val"].transform(
            lambda p: multipletests(p, method="fdr_bh")[1] if p.notna().any() else p
        )
        results_df["q_val"] = q_vals
        return results_df.sort_values(by="p_val")

    return results_df
