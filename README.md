# scVI-3D: Normalization and De-noising of Single-cell Hi-C Data

[Ye Zheng\*, Siqi Shen\* and Sündüz Keleş. Normalization and De-noising of Single-cell Hi-C Data with BandNorm and scVI-3D. bioRxiv (2021). * contribute equally.](https://www.biorxiv.org/content/10.1101/2021.03.10.434870v1)


## What is scVI-3D?

The advent of single-cell sequencing technologies in profiling 3D genome organization led to the development of single-cell high-throughput chromatin conformation (scHi-C) assays. To explicitly capture chromatin conformation features and distinguish cells based on their 3D genome organizations, we developed a fast band normalization approach, [BandNorm](https://github.com/sshen82/BandNorm), as well as a deep generative modeling framework, scVI-3D, for more structured modeling of scHi-C data. At the individual cell resolution, heterogeneity driven by the stochastic nature of chromatin fiber, various nuclear processes, and unwanted variation due to sequencing depths and batch effects poses major analytical challenges for inferring single cell-level 3D genome organizations. We develop a deep generative model, named scVI-3D, which systematically takes into account these 3D genome structural properties such as the band bias, sequencing depth effect, zero inflation, sparsity impact, and batch effects of scHi-C data.  scVI-3D is constructed based on the parametric count models of Poisson and Negative Binomial that have been successfully used in bulk measurements of chromatin conformation capture data. Single-cell resolution and the growth in the sizes of emerging scHi-C datasets have created opportunities for exploiting non-linearities in the data. Variational autoencoders offer scalable ways of learning nonlinear maps and have been successfully applied to model measurements from single cells. Motivated by the recent deep learning modeling approaches for single-cell transcription and chromatin accessibility, scVI-3D constructs the generative modeling framework on the band matrices for the dimension reduction and de-noising of scHi-C data.

Current version: 2.0

![3DVI Model](/figures/model.png)




## scVI-3D Usage

### 1. Preparation

```
git clone https://github.com/yezhengSTAT/scVI-3D
```
scVI-3D installation is finished once you successsfully git clone the repository. We provide a demo scHi-C data, sampled 400 cells of Astro, ODC, MG, Sst, four cell types from [Lee et al. 2019. Nature Methods.](https://www.nature.com/articles/s41592-019-0547-z) for test run.  The raw input data will be downloaded with this repository. In preparation for such run, you will need to have python (>=3.7) available on your server with corresponding modules required: 
  - numpy (>= 1.11.3)
  - scanpy (>= 1.4.6)
  - pandas (>= 0.21.0)
  - anndata (>= 0.7.1)
  - [scvi-tools (>= 0.8.1)](https://docs.scvi-tools.org/en/stable/installation.html)
  - joblib (>= 0.13.2)
  - scikit-learn (>= 0.21.3)
  
If you want to get the UMAP and t-SNE visualization, you will need two more modules installed:
  - matplotlib (>=3.1.1)
  - umap (0.3.10)

#### 1.1 Creating environment using ```conda``` (recommended):

- 1. Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

 - 2. Build conda environment:

```
conda env create -f scVI-3D_conda_environment.yml
```


 - 3. Active conda environment for scVI-3D:

```
conda activate schic-scvi-3d
```

To deactivate conda enviorment, use ```conda deactivate```.

#### 1.2 Install through ```pip```:

For quick python module installation, python-requirements.txt is provided in this repository. Run

```
pip install -r python-requirements.txt
```



### 2. Setting parameters for scVI-3D

    Parameters for the scVI-3D runs:
    bandMax       (-b/--bandMax)        : Maximum genomic distance to be processed, e.g. 10. Use "whole" to include all the band matrix for each chromosome. Default is "whole".
    chromList     (-c/--chromList)      : List of chromosome to be processed separate by comma, e.g. "chr1,chr2,chrX". Use "whole" to include all chromosomes in the genome size file (args.genome). Default is "whole".
    resolution    (-r/--resolution)     : Resolution of scHi-C data, e.g., 1000000.
    batchRemoval  (-br/--batchRemoval)  : Indicator to remove batch or not. Default is False. Adding '-br' to turn it on.
    nLatent       (-n/--nLatent)        : Dimension of latent space. Default is 100.
    includeDiag   (-diag/--includeDiag) : Include diagonal in the normalization and imputation process. Adding '-diag' to include. Default is not include diagonal.
    poolStrategy  (-pool/--poolStrategy): Default progressive pooling strategy combines off-diagonal 2 and 3, then 4-6, 7-10, and so on. Please refer to the Methods section of the paper for details of the pooling strategy 1, 2, 3, 4, 5. Default is 1.
    
    Path to input and output folders:
    inPath        (-i/--inPath)         : Path to the folder where input scHi-C data are saved.
    outdir        (-o/--outdir)         : Path to output directory.
    cellSummary   (-cs/--cellSummary)   : (Optional) Cell summary file with columns names to be "name" for scHi-C data file name including extension, "batch" for batch factor, "cell_type" for cluster or cell type label (tab separated file).
    genomeSize    (-g/--genome)         : Path to genome size file (tab separated file).
    save          (-s/--save)           : Save intermediate data in pickle format. Adding '-s' to turn on the saving function.
    
    Other optional parameters:
    gpuFlag       (-gpu/--gpuFlag)      : (Optional) Use GPU or not. Default is False. Adding '-gpu' to turn it on.
    parallelCPU   (-p/--parallelCPU)    : (Optional) Number of CPUs to be used for parallel running. Default is 1 and no parallel computing is used.
    pcaNum        (-pca/--pcaNum)       : (Optional) Number of principal components to be writen out. Default is 50.
    umapPlot      (-up/--umapPlot)      : (Optional) Plot UMAP of latent embeddings. Default is False. Adding '-up' to turn it on.
    tsnePlot      (-tp/--tsnePlot)      : (Optional) Plot t-SNE of latent embeddings. Default is False. Adding '-tp' to turn it on.
    verbose       (-v/--verbose)        : (Optional) Verbose. Default is False. Adding '-v' to turn it on.


### 3. Running scVI-3D

```
python3.7 Path/to/scVI-3D/scripts/scVI-3D.py -b 10 -c "whole" -r 1000000 -i "Path/to/scVI-3D/demoData" -o "Path/to/scVI-3D/results" -cs "Path/to/scVI-3D/supplementaryData/demoData_summary.txt" -g "Path/to/scVI-3D/supplementaryData/hg19.chrom.sizes" -br -n 100 -gpu -p 10 -pca 50 -up -tp -v
```

The above test run utilizes gpu and will take ~15min to finish.

### 4. Output from scVI-3D

Under output directory:

- Path/to/scVI-3D/results
    - cell_info_summary_sorted.txt: Sorted cell information matching rows of (i.e., cell) the normalization results.
    - latentEmbeddings/
      - scVI-3D_norm_PCA50.txt
      - scVI-3D_norm_latentEmbeddingFull.txt
    - scVI-3D_norm/
      - normalized count saved as a tab-separated file for each input scHi-C data with the same file name. 
      - ...
    - figures/
      - scVI-3D_norm_TSNE.pdf
      - scVI-3D_norm_UMAP.pdf
    
![UMAP Visualization](/figures/norm3DVI_UMAP.png)

![t-SNE Visualization](/figures/norm3DVI_TSNE.png)

### 5. Creating scVI-3D runs with your own data

#### 5.1 Input scHi-C data (tab separated file)

Locus-pair interactions for each cell are saved into one tab-separated file with five columns and no header indicating locus pair A (chrA binA) interacting locus pair B (chrB binB) with count N.

```
chr1    0       chr1    1000000 9
chr1    1000000 chr1    1000000 200
chr1    0       chr1    2000000 2
chr1    1000000 chr1    2000000 4
chr1    2000000 chr1    2000000 220
chr1    1000000 chr1    3000000 1
chr1    2000000 chr1    3000000 11
chr1    3000000 chr1    3000000 197
chr1    1000000 chr1    4000000 1
chr1    2000000 chr1    4000000 2
```

### 5.2 Cell summary file (tab separated file and this file is optional depending on the removal batch effect or not)

Cell summary file provides all the cell interactin file name including the extension so that we can locate the input file using the ```inPath``` parameter and the cell file name listed in column ```name```. If batch effect need to be removed, please provide the batch related information in column ```batch```. For UMAP or t-SNE visualization where cell is colored by label, please provide the cluster or cell type label in column ```cell_type```. If no batch bias removal or visualization figure generation is needed, this file can be not provided to the run. Column names need to be exactly set to ```name```, ```batch``` and ```cell_type```.


```
name    batch   cell_type       
181218_21yr_2_B11_AD001_Astro.txt       181218_21yr     Astro
181218_21yr_2_B6_AD006_Astro.txt        181218_21yr     Astro
181218_21yr_2_B6_AD007_Astro.txt        181218_21yr     Astro
181218_21yr_2_D1_AD006_Astro.txt        181218_21yr     Astro
181218_21yr_2_D6_AD008_Astro.txt        181218_21yr     Astro
181218_21yr_2_E12_AD006_Astro.txt       181218_21yr     Astro
181218_21yr_2_E6_AD010_Astro.txt        181218_21yr     Astro
181218_21yr_2_G7_AD004_Astro.txt        181218_21yr     Astro
181218_21yr_2_H9_AD001_Astro.txt        181218_21yr     Astro
181218_21yr_3_A12_AD008_Astro.txt       181218_21yr     Astro
181218_21yr_3_A9_AD008_Astro.txt        181218_21yr     Astro
181218_21yr_3_B9_AD007_Astro.txt        181218_21yr     Astro
181218_21yr_3_D5_AD008_Astro.txt        181218_21yr     Astro
181218_21yr_3_E12_AD010_Astro.txt       181218_21yr     Astro
```

### 5.3 Genome size file (tab separated file)

Here is an example of hg19 chrom size file which is tab-separated and no header. You can remove the chromosome that you do not want to process by scVI-3D for example chrY here as an instance.

```
chr1    249250621
chr2    243199373
chr3    198022430
chr4    191154276
chr5    180915260
chr6    171115067
chr7    159138663
chr8    146364022
chr9    141213431
chr10   135534747
chr11   135006516
chr12   133851895
chr13   115169878
chr14   107349540
chr15   102531392
chr16   90354753
chr17   81195210
chr18   78077248
chr19   59128983
chr20   63025520
chr21   48129895
chr22   51304566
chrX    155270560
```

### 6. Using Slurm to accelerate scVI-3D

Submit the scVI-3D run for each chromosome through ```sbatch```. Customize the sbatch command to your cluster server system.

```
chrom="chr1"
bandMax=10
resolution=1000000
inPath="Path/to/scVI-3D/demoData"
outPath="Path/to/scVI-3D/results"
cellSummary="Path/to/scVI-3D/supplementaryData/demoData_summary.txt"
genomeSize="Path/to/scVI-3D/supplementaryData/hg19.chrom.sizes"
cpuN=1

mkdir -p outLog
sbatch --nodes=1 --ntasks=1 --cpus-per-task=${cpuN} --threads-per-core=1 --gres=gpu --mem=120G --tmp=256000 -J scVI-3D_${chrom} --output=outLog/scVI-3D_${chrom}.out --export=chrom="${chrom}",bandMax="${bandMax}",resolution="${resolution}",inPath="${inPath}",outPath="${outPath}",cellSummary="${cellSummary}",genomeSize="${genomeSize}",cpuN="${cpuN}" run_scVI-3D.sh
```

In ```run_scVI-3D.sh```,

```
python3.7 Path/to/scVI-3D/scripts/scVI-3D.py -b "${bandMax}" -c "${chrom}" -r "${resolution}" -i "${inPath}" -o "${outPath}" -cs "${cellSummary}" -g "${genomeSize}" -br -n 100 -gpu -p "${cpuN}" -pca 50 -up -tp -v
```
