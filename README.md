# ChromBPNet + SuPreMo-lite in-silico mutagenesis (ISM) pipeline

This pipeline provides a modular way to run in-silico mutagenesis using tiled deletions, execute predictions using ChromBPNet and score the resulting variant effect predictions. The main scripts are:

- `make_tiled_del_vcf.py`: Generate a VCF file specifying tiled deletion variants.
- `ism_pred.py`: Run model predictions on reference and perturbed sequences.
- `ism_score.py`: Compute quantitative scores for each variant based on the model predictions.

Below are step-by-step instructions to use the pipeline.

---


## Dependencies

This pipeline requires several dependencies. Below are the main software packages and advice on installing them.

### chrombpnet

You will need the [chrombpnet](https://github.com/kundajelab/chrombpnet) model and associated code.  
- You can install it using methods described on the chrombpnet GitHub.
- This pipeline was tested using [Apptainer/Singularity](https://apptainer.org/) containers, but you may use any compatible method (e.g., Conda, Docker, or direct pip install) depending on your system and preferences.

### supremo-lite

- The pipeline uses [supremo-lite](https://github.com/gladstone-institutes/supremo_lite), which is a lightweight package for generating perturbed sequences and aligning predictions.  
- Please follow the installation instructions at the supremo-lite repository.

I used the Apptainer installation of chrombpnet and pip installed supremo-lite to the same container, and used this environment to run ism_pred.py.

To run make_tiled_del_vcf.py and ism_score.py, chromnpnet and supremo-lite are not required, and a conda environment with the required python dependencies can be used.

### Python Dependencies

The main dependencies are specified in the `pici` conda environment. See `environment.yml` in this repository for details.
The key Python packages (with version numbers) include:
  - python=3.7
  - bedtools=2.30
  - pandas=1.3
  - numpy=1.21
  - scipy=1.7
  - pysam=0.15
  - pip

**Install with conda (recommended for development):**
```bash
conda env create -f environment.yml
conda activate pici
```

Make sure to activate the conda environment or be inside the Apptainer container before running each pipeline script.


## 1. Generate Tiled Deletion Variants (`make_tiled_del_vcf.py`)

This script produces a VCF file with deletion variants tiled across specified intervals.

**Inputs:**
- BED file containing genomic intervals (`regions.bed`)
- Reference genome FASTA (`genome.fa`)
- Desired tile size (window), default = 10
- [Optional] Chromosome name format in fasta file, variant ID prefix 

**Example Command:**
```bash
python make_tiled_del_vcf.py \
    --bed regions.bed \
    --genome_fasta genome.fa \
    --out_vcf deletions.vcf \
    --tile 10 \
```
**Key Options:**
- `--tile`: Size of each deletion event (e.g., 10 bp for single-base or windowed deletions).
- `--include_partial_last_tile`: If set, includes a final deletion at the end of each interval even if the remaining sequence is shorter than the tile size.
- `--chrom_format`: Sets the chromosome naming style in the VCF output. Options are `'as_input'` (default, outputs chromosome names as they appear in the BED/region file), `'refseq'` (NC_... identifiers), or `'number'` (bare chromosome numbers or letters).
- `--id_prefix`: (Optional) String to prefix each variant ID in the VCF; useful to tag or group sets of variants.

**Output:**
- A VCF file listing all generated deletion variants.

---

## 2. Predict with ChromBPNet (`ism_pred.py`)

Next, generate model predictions for the reference and alternate/perturbed sequences using the generated VCF. The following description assumes `ism_pred.py` runs predictions for both ref/alt alleles using the model.

**Inputs:**
- Path to folder where reference genome and vcf file input are
- Reference genome FASTA (`genome.fa`), path relative to input directory
- .vcf file of variants (from step 1), path relative to input directory
- Model .h5 file

**Example Command:**
```bash
python ism_pred.py \
    --input_dir path/to/inputs \
    --vcf deletions.vcf \
    --genome_fasta genome.fa \
    --model chrombpnet_model.h5 \
    --output ism_outputs
```

**Outputs:**
- ism_outputs folder will contain:
  - Reference (unaltered) predictions, directly from ChromBPNet (.npy)
  - Reference (unaltered) predictions, aligned by base-pair by SuPreMo-lite (.npy)
  - Alternate (perturbed) predictions, directly from ChromBPNet (.npy)
  - Alternate (perturbed) predictions, aligned by base-pair by SuPreMo-lite (.npy)
  - Metadata to match predictions to variants (.csv)
  - Information about variants (.csv)

---

## 3. Score ISM Effects (`ism_score.py`)

Quantitatively summarize the prediction differences across all variants.

**Inputs:**
- `--ref_aligned`: Path to reference (unaltered) aligned predictions (.npy) from step 2
- `--alt_aligned`: Path to alternate (perturbed) aligned predictions (.npy) from step 2
- `--metadata`: Path to metadata .csv file produced in step 2 (`*_metadata.csv`)
- `--regions_bed`: BED file of regions to map variants (for region-level summaries)
- `--out_prefix`: Output prefix/path for scoring summary table (.csv will be appended)

**Example Command:**
```bash
python ism_score.py \
    --ref_aligned ref_predictions_aligned.npy \
    --alt_aligned alt_predictions_aligned.npy \
    --metadata vcf_metadata.csv \
    --regions_bed regions.bed \
    --out_prefix ism_scores_outpath/ism_scores
```

**Results:**
- `ism_scores.csv`: Tabular summary with columns such as:
  - `MSE`: Mean squared error between reference and alt predictions
  - `Spearman`/`Pearson`: Correlation coefficients between prediction profiles
  - `ref_total`/`alt_total`: Total predicted signal in reference/alt
  - Additional statistical summaries per variant
- `ism_scores_with_regions.csv`: Final results with scores joined to region mapping info and tile positions per variant
- `ism_variant_region_mapped.csv`: Table mapping each variant to the overlapping region, with region name and coordinates

---

## Example Workflow
```bash
# 1. Make tiled deletion VCF
# in conda environment pici
python make_tiled_del_vcf.py \
    --bed regions.bed \
    --genome_fasta genome.fa \
    --out_vcf deletions.vcf \
    --tile 10 \

# 2. Run model predictions
# within container with chrombpnet and supremo-lite installed
apptainer exec --nv \
  --env CUDA_VISIBLE_DEVICES=0 \
  --overlay /path/to/containers/chrombpnet_overlay.img:ro \
  --bind /parent_dir:/parent_dir \
  --bind /path/to/pici_code:/workspace \
  chrombpnet.sif \
  python /workspace/ism_pred.py \
    --input_dir ./ \
    --vcf deletions.vcf \
    --ref genome.fa \
    --model chrombpnet_model.h5 \
    --outdir ism_outputs

# 3. Score ISM predictions
# in conda environment pici
python ism_score.py \
    --ref_aligned ism_outputs/deletions_ref_predictions_aligned.npy \
    --alt_aligned ism_outputs/deletions_alt_predictions_aligned.npy \
    --metadata ism_outputs/deletions_metadata.csv \
    --regions_bed regions.bed \
    --out_prefix ism_outputs/ism_scores
```

## Troubleshooting

- Ensure all commands are run in the correct Python environment or container and that you have all dependencies.
- Double check file/folder path inputs. Some inputs require directories and some require prefixes.
- For further details, consult each script's comments or help message.

---