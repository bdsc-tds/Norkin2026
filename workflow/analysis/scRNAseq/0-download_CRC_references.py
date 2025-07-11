import os
import subprocess
import sys

sys.path.append("../../scripts")
import readwrite


def download_gse236581_files(base_outdir):
    geo_subdir = os.path.join(base_outdir, "GEO_GSE236581")
    os.makedirs(geo_subdir, exist_ok=True)

    geo_url = "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE236nnn/GSE236581/suppl/"

    print(f"Downloading GEO supplementary files into {geo_subdir} ...")

    wget_cmd = ["wget", "-r", "-np", "-nH", "--cut-dirs=5", "-P", geo_subdir, geo_url]
    subprocess.run(wget_cmd, check=True)

    print("\n✅ Finished downloading all files for GEO_GSE236581.")


def download_gse178341_files(base_outdir):
    """
    Downloads supplementary files for GEO accession GSE178341.

    This dataset corresponds to the Broad Institute's SCP1162 but is hosted
    on GEO with a different file structure (.h5 and .csv.gz).
    """
    # Use the GEO accession for the directory name for clarity
    outdir = os.path.join(base_outdir, "GEO_GSE178341")
    os.makedirs(outdir, exist_ok=True)
    print(f"Created output directory: {outdir}")

    # URLs from the GEO page for GSE178341
    base_url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE178nnn/GSE178341/suppl/"

    files_to_download = {
        "GSE178341_crc10x_full_c295v4_submit.h5": os.path.join(base_url, "GSE178341_crc10x_full_c295v4_submit.h5"),
        "GSE178341_crc10x_full_c295v4_submit_cluster.csv.gz": os.path.join(
            base_url, "GSE178341_crc10x_full_c295v4_submit_cluster.csv.gz"
        ),
        "GSE178341_crc10x_full_c295v4_submit_metatables.csv.gz": os.path.join(
            base_url, "GSE178341_crc10x_full_c295v4_submit_metatables.csv.gz"
        ),
    }

    print("--- Starting download of GSE178341 files ---")
    for filename, url in files_to_download.items():
        output_path = os.path.join(outdir, filename)
        if os.path.exists(output_path):
            print(f"File already exists, skipping: {filename}")
            continue

        print(f"Downloading {filename} from {url}...")
        try:
            # Using curl is simple and effective. -L handles redirects.
            subprocess.run(["curl", "-L", url, "-o", output_path], check=True)
            print(f"Successfully downloaded {filename}")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to download {filename}. Error: {e}")
            # Optionally, clean up partially downloaded file
            if os.path.exists(output_path):
                os.remove(output_path)
            # Stop the script if a download fails
            return

    print("\n✅ Finished downloading all files for GSE178341.")


def download_marteau2024_file(base_outdir):
    atlas_subdir = os.path.join(base_outdir, "Marteau2024")
    os.makedirs(atlas_subdir, exist_ok=True)

    atlas_url = "https://apps-01.i-med.ac.at/resources/tmp/core_atlas-adata.h5ad"
    output_file = os.path.join(atlas_subdir, "core_atlas-adata.h5ad")

    print(f"Downloading Single-cell Colorectal Cancer Atlas file to {output_file} ...")
    subprocess.run(["curl", "-L", atlas_url, "-o", output_file], check=True)

    print("Finished downloading core atlas file.")


def main():
    cfg = readwrite.config()
    base_outdir = cfg["scrnaseq_references_dir"]
    os.makedirs(base_outdir, exist_ok=True)

    download_gse236581_files(base_outdir)
    download_gse178341_files(base_outdir)
    download_marteau2024_file(base_outdir)


if __name__ == "__main__":
    main()
