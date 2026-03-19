# Organoid analysis
This repository contains the analysis presented in [Norkin et al., 2026](https://www.biorxiv.org/content/10.64898/2026.02.10.704651v2.full.pdf) downstream of the [xenium preprocessing pipeline](https://github.com/bdsc-tds/xenium_analysis_pipeline/tree/main).

## Environment Setup
### 1. Install Pixi

Install [Pixi](https://prefix.dev/docs/pixi) for environment management:

```bash
curl -sSf https://pixi.sh/install.sh | bash
```

Then restart your terminal or run:
```bash
source ~/.bashrc
```

### 2. Install environments
```bash
cd norkin_organoid
pixi config set --local run-post-link-scripts insecure
pixi install
```

You can add the environment as a Jupyter kernel named `norkin-organoid` with:
```bash
pixi run add-kernels
```

You can also activate the environments from a terminal:
```bash
pixi shell -e norkin-organoid
pixi shell -e cellcharter
```

## Repository Structure
*   **`data/`**: Directories for `xenium` and `scRNAseq` raw/processed datasets and associated metadata.
*   **`config/`**: Snakemake pipeline paths defined in `config.yml`.
*   **`workflow/`**: Snakemake logic, including `rules/` and `scripts/`. 
*   **`workflow/notebooks/`**: Analysis notebooks that were not incorporated as snakemake rules:
*   **`results/`**: Analysis outputs.
*   **`figures/`**: Figure outputs and associated .csv data.
*   **`figures_manuscript/`**: Figures symlinks into paper figures organized by figure number.
*   **`figures_manuscript_data/`**: Figures data symlinks into paper figures organized by figure number.

## Rerunning analyses
The `xenium_analysis_pipeline` snakemake pipeline mentioned at the top of this README has to be run to obtain segmentation and cell type annotation results.
All downstream analyses from the paper are in `workflow/notebooks/`.
The snakemake workflow within this repository only contains some accessory rules to generate Seurat objects of the xenium data and embedding plots. This can be launched in an HPC environment using 
```bash
./run.slurm
```
