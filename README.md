# Organoid analysis

This project uses [Pixi](https://prefix.dev/docs/pixi) for environment management and aims to provide a reproducible setup for working with the `spatialdata` ecosystem.

## 🧱 Environment Setup
### 1. Install Pixi

Install Pixi using the recommended shell script:

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

### 2. Activate environment
You can add the environment as a Jupyter kernel named `norkin-organoid` with:
```bash
pixi run add-kernel
```

You can also activate the environment from a terminal:
```bash
pixi shell -e norkin-organoid
```
