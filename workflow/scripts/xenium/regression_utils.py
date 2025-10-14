import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm  # For a nice progress bar
import warnings


import anndata
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm
import warnings


def regress_gene_morphology(
    adata: anndata.AnnData,
    morphology_features: list,
    batch_key: str = None,
    model_type: str = "ols",
    morphology_obsm_key: str = "morphology",
    use_raw: bool = False,
) -> pd.DataFrame:
    """
    Tests for association between gene expression and multiple continuous morphology
    features using a specified statistical model, with optional batch correction.

    Args:
        adata (anndata.AnnData):
            AnnData object with expression data.
        morphology_features (list):
            A list of strings with the names of morphology features to test.
        batch_key (str, optional):
            The column name in `adata.obs` that identifies the sample batch.
            If None, no batch correction is performed. Defaults to None.
        model_type (str, optional):
            The statistical model to use.
            - 'ols': Ordinary Least Squares (Linear Model). For continuous, log-transformed data. (Default)
            - 'nb': Negative Binomial GLM. For raw, integer count data with overdispersion.
            - 'poisson': Poisson GLM. For raw, integer count data where variance equals the mean.
        morphology_obsm_key (str, optional):
            Key in `adata.obsm` where morphology data is stored. Defaults to 'morphology'.
        use_raw (bool, optional):
            If True, use `adata.raw.X`. Otherwise, use `adata.X`. Defaults to False.

    Returns:
        pd.DataFrame:
            A tidy DataFrame containing test results for every gene-feature combination.
    """
    # 1. --- Parameter Validation and Setup ---
    if model_type not in ["ols", "nb", "poisson"]:
        raise ValueError("`model_type` must be either 'ols', 'nb', or 'poisson'.")

    model_names = {"ols": "Ordinary Least Squares", "nb": "Negative Binomial GLM", "poisson": "Poisson GLM"}
    print(f"--- Starting Gene-Morphology Regression ({model_names[model_type]}) ---")
    print(f"  - Features to test: {', '.join(morphology_features)}")

    # 2. --- Prepare the metadata (exog variables) ---
    if morphology_obsm_key not in adata.obsm:
        raise KeyError(f"Morphology key '{morphology_obsm_key}' not found in adata.obsm.")
    morph_df = adata.obsm[morphology_obsm_key]
    if not isinstance(morph_df, pd.DataFrame):
        print(f"  - Converting '{morphology_obsm_key}' to a pandas DataFrame...")
        morph_df = pd.DataFrame(morph_df, columns=morphology_features, index=adata.obs_names)

    # Conditionally build the sample description based on batch_key
    if batch_key:
        print(f"  - Controlling for Batch: '{batch_key}'")
        if batch_key not in adata.obs.columns:
            raise KeyError(f"Batch key '{batch_key}' not found in adata.obs.")
        sample_description = pd.concat([adata.obs[[batch_key]], morph_df[morphology_features]], axis=1)
        sample_description[batch_key] = sample_description[batch_key].astype("category")
    else:
        print("  - No batch correction applied.")
        sample_description = morph_df[morphology_features].copy()

    # 3. --- Data Selection and Safety Check ---
    expression_data = adata.raw.X if use_raw else adata.X
    response_variable = "gene_counts" if model_type in ["nb", "poisson"] else "gene_expression"

    if model_type in ["nb", "poisson"]:
        is_integer = np.all(np.equal(np.mod(expression_data[:1000].toarray(), 1), 0))
        if not is_integer:
            warnings.warn(
                f"Warning: You selected the '{model_type}' model, but the data does not appear to be "
                "integer counts. Results may be unreliable."
            )

    # 4. --- Iterate over each feature and each gene ---
    all_results = []
    for feature in tqdm(morphology_features, desc="Testing morphology features"):
        # Dynamically build the formula
        if batch_key:
            formula = f"{response_variable} ~ {batch_key} + {feature}"
        else:
            formula = f"{response_variable} ~ {feature}"

        for i, gene in enumerate(tqdm(adata.var_names, desc=f"Fitting models for {feature}", leave=False)):
            model_df = sample_description.copy()
            model_df[response_variable] = expression_data[:, i].toarray().flatten()

            try:
                if model_type == "nb":
                    model = smf.glm(formula=formula, data=model_df, family=sm.families.NegativeBinomial()).fit()
                elif model_type == "poisson":
                    model = smf.glm(formula=formula, data=model_df, family=sm.families.Poisson()).fit()
                else:  # ols
                    model = smf.ols(formula=formula, data=model_df).fit()

                all_results.append(
                    {
                        "gene": gene,
                        "morphology_feature": feature,
                        "p_val": model.pvalues[feature],
                        "coefficient": model.params[feature],
                        "std_err": model.bse[feature],
                    }
                )
            except Exception:
                all_results.append(
                    {
                        "gene": gene,
                        "morphology_feature": feature,
                        "p_val": np.nan,
                        "coefficient": np.nan,
                        "std_err": np.nan,
                    }
                )

    # 5. --- Compile results and perform multiple testing correction ---
    if not all_results:
        print("Warning: No results were generated.")
        return pd.DataFrame()

    results_df = pd.DataFrame(all_results).dropna()
    if results_df.empty:
        print("Warning: All models failed or produced NaN values.")
        return pd.DataFrame()

    results_df["q_val"] = multipletests(results_df["p_val"], method="fdr_bh")[1]
    results_df = results_df.sort_values("q_val")

    print("--- Test complete. ---")
    return results_df


def _create_synthetic_anndata(n_obs=1000, n_vars=500, n_batches=3, n_morph=5) -> (anndata.AnnData, dict):
    """
    Creates a synthetic AnnData object with known ground truth associations.
    """
    print("--- Creating synthetic dataset ---")
    print(f"  - Cells: {n_obs}, Genes: {n_vars}, Batches: {n_batches}")

    # --- A. Create base metadata ---
    batches = [f"Batch{i + 1}" for i in range(n_batches)]
    obs_df = pd.DataFrame(
        {"sample_batch": np.random.choice(batches, size=n_obs)}, index=[f"cell_{i}" for i in range(n_obs)]
    )

    morph_feature_names = [f"morph_{i}" for i in range(n_morph)]
    morph_data = pd.DataFrame(
        np.random.rand(n_obs, n_morph), index=[f"cell_{i}" for i in range(n_obs)], columns=morph_feature_names
    )

    # --- B. Create baseline raw counts with batch effects ---
    # Use a negative binomial distribution for realistic counts
    # Each batch will have a different baseline expression level (mu)
    counts = np.zeros((n_obs, n_vars), dtype=int)
    batch_baselines = {"Batch1": 2, "Batch2": 4, "Batch3": 6}

    for batch_name, base_mu in batch_baselines.items():
        mask = obs_df["sample_batch"] == batch_name
        n_batch_cells = mask.sum()
        # n=dispersion, p=probability. mu = n * (1-p)/p
        p = base_mu / (base_mu + 1)
        counts[mask, :] = np.random.negative_binomial(n=1, p=p, size=(n_batch_cells, n_vars))

    # --- C. Define and inject ground truth signals ---
    ground_truth = {
        ("gene_10", "morph_1"): {"coeff": 25.0, "desc": "Strong positive"},
        ("gene_25", "morph_2"): {"coeff": -20.0, "desc": "Strong negative"},
        ("gene_150", "morph_1"): {"coeff": -15.0, "desc": "Moderate negative"},
    }
    print("\nInjecting ground truth signals:")

    for (gene_name, morph_name), params in ground_truth.items():
        gene_idx = int(gene_name.split("_")[1])
        coeff = params["coeff"]
        print(f"  - Associating '{gene_name}' with '{morph_name}' (coeff={coeff})")

        # The signal is the morphology value scaled by the coefficient
        signal = (morph_data[morph_name] * coeff).astype(int)

        # Add the signal to the baseline counts
        counts[:, gene_idx] = np.maximum(0, counts[:, gene_idx] + signal)

    # --- D. Assemble the AnnData object ---
    adata = anndata.AnnData(X=scipy.sparse.csr_matrix(counts), obs=obs_df)
    adata.var_names = [f"gene_{i}" for i in range(n_vars)]
    adata.obsm["morphology"] = morph_data

    print("--- Synthetic dataset created. ---\n")
    return adata, ground_truth


# =============================================================================
# 2. HELPER FUNCTIONS FOR TESTING
# =============================================================================


def _create_synthetic_anndata_for_testing(n_cells=2000, n_genes=100, n_organoids=20, n_samples=4):
    """Creates a synthetic AnnData object with a known hierarchical structure and ground truth."""
    print("--- Creating synthetic dataset for validation ---")
    adata = sc.datasets.blobs(n_observations=n_cells, n_variables=n_genes)
    adata.X = np.abs(adata.X * 10).astype(int)  # Make into integer counts

    # Create hierarchical metadata: cells -> organoids -> samples
    adata.obs["organoid_id"] = [f"Org_{i}" for i in np.random.randint(0, n_organoids, n_cells)]
    org_to_sample = {f"Org_{i}": f"Sample_{i % n_samples}" for i in range(n_organoids)}
    adata.obs["sample_batch"] = adata.obs["organoid_id"].map(org_to_sample)

    # Create organoid-level morphology
    organoid_meta = pd.DataFrame(
        {"area": np.random.uniform(0.1, 1, n_organoids), "circularity": np.random.uniform(0.1, 1, n_organoids)},
        index=[f"Org_{i}" for i in range(n_organoids)],
    )
    adata.obsm["morphology"] = organoid_meta.loc[adata.obs["organoid_id"]].set_index(adata.obs.index)

    # Inject ground truth signals on top of baseline counts
    ground_truth = {
        ("gene_10", "area"): {"coeff": 50.0, "desc": "Strong positive"},
        ("gene_25", "circularity"): {"coeff": -40.0, "desc": "Strong negative"},
    }
    print("Injecting ground truth signals:")
    for (gene, morph), params in ground_truth.items():
        gene_idx = adata.var_names.get_loc(gene)
        signal = (adata.obsm["morphology"][morph] * params["coeff"]).astype(int)
        adata.X[:, gene_idx] = np.maximum(0, adata.X[:, gene_idx] + signal)
        print(f"  - Associating '{gene}' with '{morph}'")

    return adata, ground_truth


def _validate_results(results_df, ground_truth, model_type):
    """Checks if the results dataframe correctly identified the ground truth."""
    if results_df.empty:
        print("  ❌ FAILURE: Results DataFrame is empty. All models may have failed.")
        return False

    top_hits = results_df.head(10).set_index(["gene", "morphology_feature"])
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
            elif "p_direction" in hit.index:  # Bambi output
                p_dir = hit["p_direction"]
                is_significant = p_dir > 0.99 or p_dir < 0.01
                sign_ok = np.sign(hit["coefficient"]) == np.sign(params["coeff"])
                metric = f"p_direction: {p_dir:.3f}"

            if is_significant and sign_ok:
                print(f"  ✅ SUCCESS: Found in top hits with correct sign ({metric})")
            else:
                print(f"  ❌ FAILURE: Found but significance/sign was incorrect ({metric})")
                all_ok = False
        else:
            print("  ❌ FAILURE: Ground truth pair not found in top 10 significant results.")
            all_ok = False
    return all_ok


# =============================================================================
# 3. THE MAIN TEST SUITE
# =============================================================================


def validate_regression_framework():
    """Runs a suite of tests on the unified regression function."""
    adata_synth, ground_truth = _create_synthetic_anndata_for_testing()

    # Best practice: Filter lowly expressed genes before running models
    adata_synth.raw = adata_synth
    sc.pp.filter_genes(adata_synth, min_cells=10)
    sc.pp.normalize_total(adata_synth, target_sum=1e4)
    sc.pp.log1p(adata_synth)

    features_to_test = list(adata_synth.obsm["morphology"].columns)

    # Define the test scenarios
    test_scenarios = [
        {
            "name": "statsmodels GLM (Negative Binomial on Raw Counts)",
            "params": {"backend": "statsmodels", "family": "negativebinomial", "random_effects": None, "use_raw": True},
        },
        {
            "name": "statsmodels LMM (Gaussian on Log-Data, Organoid Random Effect)",
            "params": {
                "backend": "statsmodels",
                "family": "gaussian",
                "random_effects": ["organoid_id"],
                "use_raw": False,
            },
        },
        {
            "name": "Bambi GLMM (Negative Binomial on Raw Counts, Nested Random Effects)",
            "params": {
                "backend": "bambi",
                "family": "negativebinomial",
                "random_effects": ["organoid_id", "sample_batch"],
                "use_raw": True,
            },
        },
    ]

    # Run all tests
    for scenario in test_scenarios:
        print("\n" + "=" * 50)
        print(f"RUNNING TEST: {scenario['name']}")
        print("=" * 50)

        results = regress_gene_morphology(adata=adata_synth, morphology_features=features_to_test, **scenario["params"])

        is_successful = _validate_results(results, ground_truth, scenario["params"]["backend"])

        if is_successful:
            print(f"\nCONCLUSION for {scenario['name']}: ✅ PASSED")
        else:
            print(f"\nCONCLUSION for {scenario['name']}: ❌ FAILED")


# --- Run the validation suite ---
if __name__ == "__main__":
    validate_regression_framework()
