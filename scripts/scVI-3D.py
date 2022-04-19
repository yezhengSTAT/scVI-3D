#!/usr/bin/env python

## scVI-3D
## Author: Ye Zheng
## Contact: yezheng.stat@gmail.com


## Script to remove sequencing depth effect, batch effect and lower dimension project, denoising single-cell Hi-C data.
## APRIL 2022

import sys
import os
import glob
import gc
import argparse
import pickle
from tqdm import tqdm
import scanpy as sc
import numpy as np
import pandas as pd
import anndata
import scvi
from joblib import Parallel, delayed
from sklearn.decomposition import PCA

def create_band_mat(x: np.array, count: np.array, diag: int, maxChromosomeSize: int) -> np.array:
    bandMat = np.zeros(maxChromosomeSize - diag)
    bandMat[x] = count
    return bandMat


class Process(object):
    def __init__(self, resolution, chromSize=None):
        self._RESOLUTION = resolution
        self._chromSize = chromSize
        self.df = None
        self._lastchrom = None
        self._chormdf = None
    
    def rescale(self, chrA, x, y, counts, resolution = None):
        if resolution:
            self._RESOLUTION = resolution
        
        xR = x // self._RESOLUTION
        yR = y // self._RESOLUTION
        self.df = pd.DataFrame({'chrA': chrA,
                        'x': xR,
                        'y': yR,
                        'counts': counts})
        self.df.loc[:,'diag'] = abs(yR - xR)
        return True
    
    def band(self, chrom, diag, maxBand):
        if self.df is None:
            raise "Run process.rescale(chrA, binA, binY, counts, resolution) first."        
        
        if self._lastchrom is None or (self._lastchrom != chrom):
            self._lastchrom = chrom
            self._chormdf = self.df[self.df.chrA == chrom]            
        
        dat =  self._chormdf[self._chormdf.diag == diag]
        mat = create_band_mat(dat.x.values, dat.counts.values, diag, maxBand)
        return mat
    
    def band_all(self, chromSize, used_chroms = 'whole', used_diags = [i for i in range(1, 11)]):
        if self.df is None:
            raise "Run process.rescale(chrA, binA, binY, counts, resolution) first"
        if chromSize:
            self._chromSize = chromSize
            
        chrom = 'chrA'
        diag_s = 'diag'
                
        cell_band = {}
        for chromosome, chromosome_data in self.df.groupby(chrom):
            if (used_chroms != 'whole' and chromosome not in used_chroms) or chromosome not in self._chromSize:
                continue
            
            bandSize = self._chromSize[chromosome] // self._RESOLUTION + 1
            chromosome_band = {}
            for diag, chromosome_diag in chromosome_data.groupby(diag_s):
                if used_diags != 'whole' and diag not in used_diags:
                    continue
                x = chromosome_diag.x.values
                count = chromosome_diag.counts.values
                chromosome_band[diag] = create_band_mat(x, count, diag, bandSize)
            cell_band[chromosome] = chromosome_band
        return cell_band
    
def read_file(file):
    df = pd.read_csv(file, sep = "\t", header = None, names = ['chrA', 'binA', 'chrB', 'binB', 'counts'])
    df.loc[:,'cell'] = file
    return df

def read_file_chrom(file, used_chroms):
    dfTmp = pd.read_csv(file, sep = "\t", header = None, names = ['chrA', 'binA', 'chrB', 'binB', 'counts'])
    dfTmp.loc[:,'cell'] = file    
    if used_chroms == 'whole':
        df = dfTmp
    else:
        df = dfTmp[dfTmp.chrA.isin(used_chroms)]

    return df

def read_files(file_list, used_chroms = 'whole', cores = 8):

    df_list = Parallel(n_jobs=cores)(delayed(read_file_chrom)(file, used_chroms) for file in file_list)
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df
    
def process_cell(cell_raw: pd.DataFrame, binSizeDict: dict, 
                 resolution: int, cell: str,
                 used_chroms,
                 used_diags):
    process = Process(resolution)
    process.rescale(cell_raw.chrA, cell_raw.binA, cell_raw.binB, cell_raw.counts)    
    cell_band = process.band_all(binSizeDict, used_chroms=used_chroms, used_diags=used_diags)
    return (cell_band, cell)



def get_locuspair(imputeM, chromSelect, bandDist):
    xvars = ['chrA','binA', 'chrB', 'binB', 'count', 'cellID']
    tmp = imputeM.transpose().copy()
    tmp.index.name = 'binA'
    normCount = pd.melt(tmp.reset_index(), id_vars = ['binA'], var_name='cellID', value_name='count')
    normCount.loc[:,'binA'] = normCount.binA.astype(int)
    normCount.loc[:,'binB'] = normCount.binA + bandDist

    normCount.loc[:,'chrA'] = chromSelect
    normCount.loc[:,'chrB'] = chromSelect
    normCount = normCount[xvars]
    
    return normCount


def normalize(bandM, cellInfo, chromSelect, bandDist, nLatent = 100, batchFlag = False, gpuFlag = False):

    #bandM = band_chrom_diag[chromSelect][bandDist] #pd.read_csv(args.infile, index_col = 0).round(0)
    cellSelect = [i for i, val in enumerate(bandM.sum(axis = 1)>0) if val]
    if len(cellSelect) == 0:
        normCount = None
        latentDF = pd.DataFrame(np.zeros((len(bandM), nLatent)), index = range(len(bandM)))
        
    else:
        bandDepth = bandM[cellSelect,].sum(axis = 1).mean()
        adata = sc.AnnData(bandM)
    
        if(batchFlag is True):
            adata.obs['batch'] = cellInfo['batch'].values
    
        sc.pp.filter_cells(adata, min_counts=1)

        if(batchFlag is True):
            scvi.data.setup_anndata(adata, batch_key = 'batch')
        else:
            scvi.data.setup_anndata(adata)
        model = scvi.model.SCVI(adata, n_latent = nLatent, use_cuda = gpuFlag)
    
        model.train()

        if(batchFlag is True):
            imputeTmp = np.zeros((len(cellSelect), bandM.shape[1]))
            for batchName in list(set(cellInfo['batch'].values)):
                imputeTmp = imputeTmp + model.get_normalized_expression(library_size = bandDepth, transform_batch = batchName)
            imputeM = imputeTmp/len(list(set(cellInfo['batch'].values)))

        else:
            imputeM = model.get_normalized_expression(library_size = bandDepth)
        
        normCount = get_locuspair(imputeM, chromSelect, bandDist)

        latent = model.get_latent_representation()
        latentDF = pd.DataFrame(latent, index = cellSelect)
        latentDF = latentDF.reindex([i for i in range(len(bandM))]).fillna(0)

    return(latentDF, normCount)



def get_args():
    '''Get arguments'''
    parser = argparse.ArgumentParser(description = '------------Usage Start------------',
                                     epilog = '------------Usage End------------')
    parser.add_argument('-b', '--bandMax', help = 'Maximum genomic distance to be processed, e.g. 10. Use "whole" to include all the band matrix for each chromosome. Default is "whole".', default = 'whole')
    parser.add_argument('-c', '--chromList', help = 'List of chromosome to be processed separate by comma, e.g. "chr1,chr2,chrX". Use "whole" to include all chromosomes in the cell summary file (args.cellSummary). Default is "whole".', default = 'whole')
    parser.add_argument('-r', '--resolution', help = 'Resolution of scHi-C data, e.g., 1000000.', default = None)
    parser.add_argument('-br', '--batchRemoval', help = 'Indicator to remove batch or not. Default is False.', action='store_true')
    parser.add_argument('-n', '--nLatent', help = 'Dimension of latent space. Default is 100.', default = 100)
    parser.add_argument('-diag', '--includeDiag', help = 'Include diagonal in the normalization and imputation process. Default is not include.', action='store_true')
    parser.add_argument('-pool', '--poolStrategy', help = 'Pooling strategy: 1, 2, 3, 4, 5.', default = 1)
    parser.add_argument('-i', '--inPath', help = 'Path to the folder where input scHi-C data are saved.', default = None)
    parser.add_argument('-o', '--outdir', help = 'Path to output directory.', default = None)
    parser.add_argument('-cs', '--cellSummary', help = '(Optional) Cell summary file with columns names to be "name" for scHi-C data file name including extension, "batch" for batch factor, "cell_type" for cluster or cell type label (tab separated file).', default = None)
    parser.add_argument('-g', '--genome', help = 'Path to genome size file (tab separated file).', default = None)
    parser.add_argument('-s', '--save', help = '(Optional) Save intermediate data in pickle format.', action = 'store_true')
    parser.add_argument('-gpu', '--gpuFlag', help = '(Optional) Use GPU or not. Default is False.', action='store_true')
    parser.add_argument('-p', '--parallelCPU', help = '(Optional) Number of CPUs to be used for parallel running. Default is 1 and no parallel computing is used.', default = 1)
    parser.add_argument('-pca', '--pcaNum', help = '(Optional) Number of principal components to be writen out. Default is 50.', default = 50)
    parser.add_argument('-up', '--umapPlot', help = '(Optional) Plot UMAP of latent embeddings. Default is False.', action='store_true')
    parser.add_argument('-tp', '--tsnePlot', help = '(Optional) Plot t-SNE of latent embeddings. Default is False.', action='store_true')
    parser.add_argument('-v', '--verbose', help = '(Optional) Verbose. Default is False.', action='store_true')

    
    args = parser.parse_args()
    
    if args.bandMax != "whole":
        if int(args.bandMax) <= 0:
            print("Maximum distance as positive integer for band matrix need to be specified.")
            parser.print_help()
            sys.exit()
    
    if args.resolution is None:
        print("Please provide the resolution of the data.")
        parser.print_help()
        sys.exit()
    if args.inPath is None:
        print("Path to the input scHi-C data need to be specified.")
        parser.print_help()
        sys.exit()
    if args.outdir is None:
        print("Path to output directory need to be specified.")
        parser.print_help()
        sys.exit()
    else:
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
            
    if args.batchRemoval is True:
        if not os.path.exists(args.cellSummary):
            print("Cell summary file with batch information does not exist.")
            parser.print_help()
            sys.exit()
        cellInfo = pd.read_csv(args.cellSummary, sep = "\t", header = 0)
        if 'batch' not in cellInfo.columns:
            print("There is no column in cell summary file called 'batch'.")
            parser.print_help()
            sys.exit()
    
    if not os.path.exists(args.genome):
        print("The genome size file does not exist.")
        parser.print_help()
        sys.exit()
        
    if int(args.nLatent) <= 0:
        print("The number of dimension of latent space need to be set as nonnegative integer.")
        parser.print_help()
        sys.exit()
        
    if int(args.poolStrategy) not in [1, 2, 3, 4, 5, 6, 7]:
        print("Please select one pooling strategy from 1, 2, 3, 4, 5. Default is 1.")
        parser.print_help()
        sys.exit()
    
    if int(args.parallelCPU) == -1:
        print("All the CPUs will be used.")
    if int(args.parallelCPU) == -2:
        print("All the CPUs but one will be used.")
    if int(args.parallelCPU) < -2:
        print("Please provide number of CPUs for parallel computing.")
        parser.print_help()
        sys.exit()
        
    if int(args.pcaNum) < 2:
        print("Please provide a number larger than 1 for PCA results writing out.")
        parser.print_help()
        sys.exit()
    return args

if __name__ == "__main__":

    args = get_args()
    
    if args.verbose:
        print('Maximum genomic distance:', args.bandMax)
        print('Chromosomes to be processed:', args.chromList)
        print('Resolution:', args.resolution)
        print('Remove batch effect or not:', args.batchRemoval)
        print('Dimension of latent space:', args.nLatent)
        print('Include diagonal:', args.includeDiag)
        print('Pooling strategy:', args.poolStrategy)
        print('Path to the input scHi-C data:', args.inPath)
        print('Path to output directory:', args.outdir)
        print('Cell summary file:', args.cellSummary)
        print('Genome size file:', args.genome)
        print('Save intermediate data:', args.save)
        print('Use GPU or not:', args.gpuFlag)
        print('Number of CPUs for parallel computing:', args.parallelCPU)
        print('PCA number:', args.pcaNum)
        print('UMAP plot:', args.umapPlot)
        print('t-SNE plot:', args.tsnePlot)
   
    
    outdir = args.outdir
    saveFlag = args.save
    if saveFlag:
        if not os.path.exists(outdir + '/pickle'):
            os.mkdir(outdir + '/pickle')

    ## number of bin per chromosome
    print("Caculate total number of bin per chromosome.")
    binSize = pd.read_csv(args.genome, sep = "\t", header = None)

    binSizeDict = {}
    N = binSize.shape[0]
    for i in range(N):
        chrome = binSize.iloc[i,0]
        size = binSize.iloc[i,1]
        binSizeDict[chrome] = size
    
    ## cell info file
    print("Prepare cell summary file.")
    if args.cellSummary is not None:
        cellInfo = pd.read_csv(args.cellSummary, sep = "\t", header = 0).sort_values(by = 'name')
        pd.DataFrame(cellInfo).to_csv(outdir + '/cell_info_summary_sorted.txt', sep = '\t', header = False, index = False)
    else:
        cellName = {'name': os.listdir(args.inPath)}
        cellInfo = pd.DataFrame(cellName).sort_values(by = 'name')
        pd.DataFrame(cellInfo).to_csv(outdir + '/cell_info_summary_sorted.txt', sep = '\t', header = False, index = False)

    cellInfo.index = range(cellInfo.shape[0])
    
    ## read in scHi-C data
    print("Read in scHi-C data.")
    files = list(args.inPath + '/' + cellInfo.name)
    files.sort()
    
    ## read all the files and sort by cell file name
    resolution = int(args.resolution)
    coreN = int(args.parallelCPU)
    if args.bandMax == "whole":
        used_diags = "whole"
    else:
        if args.includeDiag:
            used_diags = [i for i in range(0, int(args.bandMax) + 1)]
        else:
            used_diags = [i for i in range(1, int(args.bandMax) + 1)]
    
    if args.chromList == "whole":
        used_chroms = "whole"
    else:
        used_chroms = args.chromList.split(',')
    
    raws = read_files(files, used_chroms, coreN)
    
    print("Convert interactions into band matrix.")
    # raw_cells = Parallel(n_jobs=coreN)(delayed(process_cell)(cell_df, binSizeDict, resolution, cell, used_chroms, used_diags) for cell, cell_df in tqdm(raws.groupby('cell')))
    if coreN == 1:
        raw_cells = [process_cell(cell_df, binSizeDict, resolution, cell, used_chroms, used_diags) for cell, cell_df in raws.groupby('cell')]
    else:
        raw_cells = Parallel(n_jobs=coreN)(delayed(process_cell)(cell_df, binSizeDict, resolution, cell, used_chroms, used_diags) for cell, cell_df in tqdm(raws.groupby('cell')))

    raw_cells.sort(key=lambda x: x[1]) ##x[1] is 'cell', used for sorting
    cells = [cell for _, cell in raw_cells]
    raw_cells = [raw_cell for raw_cell, _ in raw_cells]
    if saveFlag:
        with open(outdir + '/pickle/raw_cells', 'wb') as f:
            pickle.dump(raw_cells, f)

    # del raws
    # gc.collect()
    
    print("Concat cells into cell x locus-pair matrix.")
    band_chrom_diag = {}
    for chrom, chromSize in binSizeDict.items():
        if used_chroms != "whole" and chrom not in used_chroms:
            continue
        chromSize = chromSize // resolution + 1
        chrom_diag = {}
        if args.includeDiag:
            band_start = 0
        else:
            band_start = 1
        for band in range(band_start, chromSize):
            if used_diags != "whole" and band not in used_diags:
                continue
            mat = []
            for fi in range(len(files)):
                if band not in raw_cells[fi][chrom]:
                    tmp = np.zeros(chromSize - band)
                else:
                    tmp = raw_cells[fi][chrom][band]
                mat.append(tmp)
            chrom_diag[band] = np.vstack(mat)
        band_chrom_diag[chrom] = chrom_diag
    
    if saveFlag:
        with open(outdir + '/pickle/band_chrom_diag', 'wb') as f:
            pickle.dump(band_chrom_diag, f)
    
    # del raw_cells
    # gc.collect()

    ## pooling strategy 1
    
    if int(args.poolStrategy) == 1: ## 1, 1+2, 3+4+5, 6+7+8+9, ...
        print("Progressive Pooling Strategy 1 (Default).")
        chrom_band_pool = {}

        for chrom, chromSize in binSizeDict.items():
            if used_chroms != "whole" and chrom not in used_chroms:
                continue
            chromSize = chromSize // resolution + 1
            chrom_band_pool[chrom] = {}
            chrom_ind = 1
            increase_unit = 0
            pool_ind = 1
            if used_diags == "whole":
                chromMax = chromSize
            else:
                chromMax = max(used_diags)
            
            while chrom_ind < chromMax:
                start_ind = chrom_ind 
                end_ind = min(chromMax, chrom_ind + increase_unit + 1)
                if (chrom_ind + increase_unit + 1 > chromMax) and (end_ind - start_ind < 10):
                    chrom_ind = chrom_ind + increase_unit + 1
                    increase_unit = increase_unit + 1
                    pool_ind = pool_ind + 1
                else:    
                    chrom_band_pool[chrom][pool_ind] = np.arange(start_ind, end_ind)
                    chrom_ind = chrom_ind + increase_unit + 1
                    increase_unit = increase_unit + 1
                    pool_ind = pool_ind + 1

    if int(args.poolStrategy) == 2: ## save 1-10 individually, 11+12, 13+14+15, ...
        print("Progressive Pooling Strategy 2.")
        chrom_band_pool = {}

        for chrom, chromSize in binSizeDict.items():
            if used_chroms != "whole" and chrom not in used_chroms:
                continue
            chromSize = chromSize // resolution + 1
            chrom_band_pool[chrom] = {}
            ## First 10 band on its own
            for i in np.arange(1, 11):
                chrom_band_pool[chrom][i] = [i]
            ## Start merging band from 11st band
            chrom_ind = 11
            increase_unit = 1
            pool_ind = 11
            if used_diags == "whole":
                chromMax = chromSize
            else:
                chromMax = max(used_diags)
            
            while chrom_ind < chromMax:
                start_ind = chrom_ind 
                end_ind = min(chromMax, chrom_ind + increase_unit + 1)
                if (chrom_ind + increase_unit + 1 > chromMax) and (end_ind - start_ind < 10):
                    chrom_ind = chrom_ind + increase_unit + 1
                    increase_unit = increase_unit + 1
                    pool_ind = pool_ind + 1
                else:    
                    chrom_band_pool[chrom][pool_ind] = np.arange(start_ind, end_ind)
                    chrom_ind = chrom_ind + increase_unit + 1
                    increase_unit = increase_unit + 1
                    pool_ind = pool_ind + 1

    if int(args.poolStrategy) == 3: ## merge 1-5, 6-10, 11-20, 21-40, 41-70, ...
        print("Progressive Pooling Strategy 3.")
        chrom_band_pool = {}

        for chrom, chromSize in binSizeDict.items():
            if used_chroms != "whole" and chrom not in used_chroms:
                continue
            chromSize = chromSize // resolution + 1
            chrom_band_pool[chrom] = {}
            ## Merge first 10 bands 
            chrom_band_pool[chrom][1] = [1, 2, 3, 4, 5]
            chrom_band_pool[chrom][2] = [6, 7, 8, 9, 10]
            
            ## Start merging band from 11st band
            chrom_ind = 11
            increase_unit = 10
            pool_ind = 3
            if used_diags == "whole":
                chromMax = chromSize
            else:
                chromMax = max(used_diags)
            
            while chrom_ind < chromMax:
                start_ind = chrom_ind 
                end_ind = min(chromMax, chrom_ind + increase_unit + 1)
                if (chrom_ind + increase_unit + 1 > chromMax) and (end_ind - start_ind < 10):
                    chrom_ind = chrom_ind + increase_unit + 1
                    increase_unit = increase_unit + 10
                    pool_ind = pool_ind + 1
                else:    
                    chrom_band_pool[chrom][pool_ind] = np.arange(start_ind, end_ind)
                    chrom_ind = chrom_ind + increase_unit + 1
                    increase_unit = increase_unit + 10
                    pool_ind = pool_ind + 1

    if int(args.poolStrategy) == 4: ## 20 bands per unit
        print("Progressive Pooling Strategy 4.")
        chrom_band_pool = {}

        for chrom, chromSize in binSizeDict.items():
            if used_chroms != "whole" and chrom not in used_chroms:
                continue
            chromSize = chromSize // resolution + 1
            chrom_band_pool[chrom] = {}
            
            ## Start merging band from 11st band
            chrom_ind = 1
            increase_unit = 20
            pool_ind = 1
            if used_diags == "whole":
                chromMax = chromSize
            else:
                chromMax = max(used_diags)
            
            while chrom_ind < chromMax:
                start_ind = chrom_ind 
                end_ind = min(chromMax, chrom_ind + increase_unit + 1)
                if (chrom_ind + increase_unit + 1 > chromMax) and (end_ind - start_ind < 10):
                    chrom_ind = chrom_ind + increase_unit + 1
                    ## increase_unit = increase_unit + 1
                    pool_ind = pool_ind + 1
                else:    
                    chrom_band_pool[chrom][pool_ind] = np.arange(start_ind, end_ind)
                    chrom_ind = chrom_ind + increase_unit + 1
                    ## increase_unit = increase_unit + 1
                    pool_ind = pool_ind + 1

    if int(args.poolStrategy) == 5: ## 1-10, and 20 bands per unit after
        print("Progressive Pooling Strategy 5.")
        chrom_band_pool = {}

        for chrom, chromSize in binSizeDict.items():
            if used_chroms != "whole" and chrom not in used_chroms:
                continue
            chromSize = chromSize // resolution + 1
            chrom_band_pool[chrom] = {}
            ## First 10 band merged
            chrom_band_pool[chrom][1] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            ## Start merging band from 11st band
            chrom_ind = 11
            increase_unit = 20
            pool_ind = 2
            if used_diags == "whole":
                chromMax = chromSize
            else:
                chromMax = max(used_diags)
            
            while chrom_ind < chromMax:
                start_ind = chrom_ind 
                end_ind = min(chromMax, chrom_ind + increase_unit + 1)
                if (chrom_ind + increase_unit + 1 > chromMax) and (end_ind - start_ind < 10):
                    chrom_ind = chrom_ind + increase_unit + 1
                    ## increase_unit = increase_unit + 1
                    pool_ind = pool_ind + 1
                else:    
                    chrom_band_pool[chrom][pool_ind] = np.arange(start_ind, end_ind)
                    chrom_ind = chrom_ind + increase_unit + 1
                    ## increase_unit = increase_unit + 1
                    pool_ind = pool_ind + 1

    if int(args.poolStrategy) == 6: ## 1, 2+3, 4+5+6, 7+8+9+10, 11-20, 21-40, ...
        print("Progressive Pooling Strategy 6.")
        chrom_band_pool = {}

        for chrom, chromSize in binSizeDict.items():
            if used_chroms != "whole" and chrom not in used_chroms:
                continue
            chromSize = chromSize // resolution + 1
            chrom_band_pool[chrom] = {}
            ## First 10 band merged
            chrom_band_pool[chrom][1] = [1]
            chrom_band_pool[chrom][2] = [2, 3]
            chrom_band_pool[chrom][3] = [4, 5, 6]
            chrom_band_pool[chrom][4] = [7, 8, 9, 10]
            ## Start merging band from 11st band
            chrom_ind = 11
            increase_unit = 10
            pool_ind = 5
            if used_diags == "whole":
                chromMax = chromSize
            else:
                chromMax = max(used_diags)
            
            while chrom_ind < chromMax:
                start_ind = chrom_ind 
                end_ind = min(chromMax, chrom_ind + increase_unit + 1)
                if (chrom_ind + increase_unit + 1 > chromMax) and (end_ind - start_ind < 10):
                    chrom_ind = chrom_ind + increase_unit + 1
                    increase_unit = increase_unit + 10
                    pool_ind = pool_ind + 1
                else:    
                    chrom_band_pool[chrom][pool_ind] = np.arange(start_ind, end_ind)
                    chrom_ind = chrom_ind + increase_unit + 1
                    increase_unit = increase_unit + 10
                    pool_ind = pool_ind + 1

    if int(args.poolStrategy) == 7: ## similar to strategy 1 but multiple by 10
        print("Progressive Pooling Strategy 7.") ## similar to strategy 1 but enlarge by 10 times
        chrom_band_pool = {}

        for chrom, chromSize in binSizeDict.items():
            if used_chroms != "whole" and chrom not in used_chroms:
                continue
            chromSize = chromSize // resolution + 1
            chrom_band_pool[chrom] = {}
            
            ## Start merging band from 1st band
            chrom_ind = 1
            increase_unit = 10
            pool_ind = 1
            if used_diags == "whole":
                chromMax = chromSize
            else:
                chromMax = max(used_diags)
            
            while chrom_ind < chromMax:
                start_ind = chrom_ind 
                end_ind = min(chromMax, chrom_ind + increase_unit + 1)
                if (chrom_ind + increase_unit + 1 > chromMax) and (end_ind - start_ind < 10):
                    chrom_ind = chrom_ind + increase_unit + 1
                    increase_unit = increase_unit + 10
                    pool_ind = pool_ind + 1
                else:    
                    chrom_band_pool[chrom][pool_ind] = np.arange(start_ind, end_ind)
                    chrom_ind = chrom_ind + increase_unit + 1
                    increase_unit = increase_unit + 10
                    pool_ind = pool_ind + 1
                    
    print("Start pooling cell x locus-pair matrix.")
    band_chrom_diag_pool = {}
    for chrom, chromSize in binSizeDict.items():
        if used_chroms != "whole" and chrom not in used_chroms:
            continue
        chromSize = chromSize // resolution + 1
        band_chrom_diag_pool[chrom] = {}
        for pool_ind in chrom_band_pool[chrom].keys():
            # print(pool_ind)
            init = True
            for band_ind in chrom_band_pool[chrom][pool_ind]:
                if used_diags != "whole" and band_ind not in used_diags:
                    continue
                if(init == True):
                    init = False
                    band_chrom_diag_pool[chrom][pool_ind] = band_chrom_diag[chrom][band_ind]
                else:
                    band_chrom_diag_pool[chrom][pool_ind] = np.hstack((band_chrom_diag_pool[chrom][pool_ind], band_chrom_diag[chrom][band_ind]))
            # if pool_ind in band_chrom_diag_pool[chrom].keys():
            #     print(band_chrom_diag_pool[chrom][pool_ind].shape)

    if saveFlag:
        with open(outdir + '/pickle/band_chrom_diag_pool', 'wb') as f:
            pickle.dump(band_chrom_diag_pool, f)

    ## scVI-3D
    print("scVI-3D normalization.")
    bandMiter = [[bandM, chromSelect, bandDist] for chromSelect, band_diags in band_chrom_diag.items() for bandDist, bandM in band_diags.items()]
    nLatent = int(args.nLatent) #int(args.nLatent)
    batchFlag = args.batchRemoval
    gpuFlag = args.gpuFlag

    if coreN == 1:
        res = [normalize(bandM, cellInfo, chromSelect, bandDist, nLatent, batchFlag, gpuFlag) for bandM, chromSelect, bandDist in bandMiter]
    else:
        res = Parallel(n_jobs=coreN,backend='multiprocessing')(delayed(normalize)(bandM, cellInfo, chromSelect, bandDist, nLatent, batchFlag, gpuFlag) for bandM, chromSelect, bandDist in bandMiter)
    if saveFlag:
        with open(outdir + '/pickle/res', 'wb') as f:
            pickle.dump(res, f)
    
    print("Writing out latent embeddings and normalization counts.")
    
    if not os.path.exists(outdir + '/scVI-3D_norm'):
        os.mkdir(outdir + '/scVI-3D_norm')
    

    ## remove existing files
    i = 0
    for cellId, cellDf in res[i][1].groupby('cellID'):
            fname = outdir + '/scVI-3D_norm/' + cells[int(cellId)].split('/')[-1]
            if os.path.exists(fname):
                os.remove(fname)
            cellDf.drop(columns=['cellID']).to_csv(fname, sep='\t', header=False, index=False, mode='a')
    
    print('Writing out normalization count.')
    for i in range(1, len(res)):
        if res[i][1] is None:
            continue
        for cellId, cellDf in res[i][1].groupby('cellID'):
            fname = outdir + '/scVI-3D_norm/' + cells[int(cellId)].split('/')[-1]
            cellDf.drop(columns=['cellID']).to_csv(fname, sep='\t', header=False, index=False, mode='a')
    
    ## concatenate latent embeddings across band matrices
    latentList = [res[i][0] for i in range(len(res))]
    latentM = pd.concat(latentList, axis = 1)
    if not os.path.exists(outdir + '/latentEmbeddings'):
        os.mkdir(outdir + '/latentEmbeddings')
    pd.DataFrame(latentM).to_csv(outdir + '/latentEmbeddings/scVI-3D_norm_latentEmbeddingFull.txt', sep = '\t', header = False, index = False)
    
    ## obtain the PCA latent embeddings as well
    pca = PCA(n_components = int(args.pcaNum))
    latentPCA = pca.fit_transform(latentM)
    pd.DataFrame(latentPCA).to_csv(outdir + '/latentEmbeddings/scVI-3D_norm_PCA' + str(args.pcaNum) + '.txt', sep = '\t', header = False, index = False)
    
    
    ## Visualization
    if args.umapPlot:
        print("UMAP visualization.")
        if not os.path.exists(outdir + '/figures/'):
            os.mkdir(outdir + '/figures/')
        import matplotlib.pyplot as plt
        import umap
        reducer = umap.UMAP()
        latentUMAP = reducer.fit_transform(latentPCA)

        colorDict = dict(zip(set(cellInfo.cell_type), range(len(set(cellInfo.cell_type)))))
        cellLabel = [colorDict[cellInfo['cell_type'][i]] for i in range(len(cellInfo))]
 
        fig, ax = plt.subplots(1, figsize=(14, 10))
        plt.scatter(
            latentUMAP[:, 0],
            latentUMAP[:, 1],
            c = cellLabel)
        plt.setp(ax, xticks=[], yticks=[])
        cbar = plt.colorbar(boundaries=np.arange(len(list(colorDict.keys()))+1) - 0.5)
        cbar.set_ticks(np.arange(len(list(colorDict.keys()))+2))
        cbar.set_ticklabels(list(colorDict.keys()))
        plt.title('UMAP Projection of the scHi-C Demo Data', fontsize=20)
        plt.savefig(outdir + '/figures/scVI-3D_norm_UMAP.pdf')  

    if args.tsnePlot:
        print("t-SNE visualization.")
        if not os.path.exists(outdir + '/figures/'):
            os.mkdir(outdir + '/figures/')
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=0)
        latentTSNE = tsne.fit_transform(latentPCA)

        fig, ax = plt.subplots(1, figsize=(14, 10))
        plt.scatter(
            latentTSNE[:, 0],
            latentTSNE[:, 1],
            c = cellLabel)
        plt.setp(ax, xticks=[], yticks=[])
        cbar = plt.colorbar(boundaries=np.arange(len(list(colorDict.keys()))+1) - 0.5)
        cbar.set_ticks(np.arange(len(list(colorDict.keys()))+2))
        cbar.set_ticklabels(list(colorDict.keys()))
        plt.title('t-SNE Projection of the scHi-C Demo Data', fontsize=20)
        plt.savefig(outdir + '/figures/scVI-3D_norm_TSNE.pdf')  
