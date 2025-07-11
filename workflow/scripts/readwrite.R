library(Seurat)
library(yaml)
# install.packages("this.path")

script_dir = this.path::here()
config_path = paste0(script_dir, "/../../config/config.yml")

config <- function(path = config_path) {
  # Read the configuration file and return a dictionary of config values.

  # Parameters
  # ----------
  # path : str
  #     The path to the configuration file. Defaults to the value of
  #     `config_path` if not provided.

  # Returns
  # -------
  # cfg : dict
  #     A dictionary of configuration values. All values are strings and
  #     have been converted to absolute paths by prepending the value of
  #     `cfg["base_dir"]`.

  cfg <- read_yaml(path)
  for (k in names(cfg)) {
    if (k != "base_dir") {
      cfg[[k]] <- paste0(cfg$base_dir, cfg[[k]])
    }
  }
  return(cfg)
}
