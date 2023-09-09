# stVAE
stVAE, a method based on the variational autoencoder framework to deconvolve the cell-type composition of spatial transcriptomics at cellular resolution.

## Table of Contents
1. [Installation](#installation)
2. [API](#api)
3. [Data](#data)
4. [Example workflow](#example-workflow)
## Installation
### Requirements
- Python packages (3.8+)
  - scvi (0.16.1)
  - torch (1.11.0)
  - scanpy (1.9.1)
  - numpy (1.22.0)
  - pandas (1.4.2)
  - scipy (1.10.0)
  - anndata (0.8.0)
  - sparsemax (0.1.9)


### Install stVAE
Download and install stVAE
```
git clone --recursive https://github.com/lichen2018/stVAE.git
cd stVAE
python3 setup.py build
python3 setup.py install
```
## API
### Preprocess data
```python
get_cell_type_profile(sc_adata, st_adata, mu_expr_file='mu_gene_expression.csv', disper_file='disp_gene_expression.csv', scRNA_data_file='scRNA.csv', scRNA_label_file='scRNA_label.csv', spatial_data_file='stRNA.csv', n_epochs=250)
```
#### Description
  ```
  Calculate cell-type specific mean expression of genes and gene-specific dispersion parameters, generate processed single cell data and annotations used to construct pseudo spots, and generate processed spatial transcriptomic data.
  ```
#### Parameters  
  ```
  sc_adata              single cell anndata file.
  st_adata              spatial transcriptomic anndata file.
  mu_expr_file          file used to save cell-type specific mean exrepssion of genes.
  disper_file           file used to save gene dispersion.
  scRNA_data_file       file used to save processed single cell data.
  scRNA_label_file      file used to save annotation information of single cell data.
  spatial_data_file     file used to save processed spatial transcriptomics.
  n_epochs              number of epochs to esimate cell-type specific mean expression and dispersion parameters of genes.
  ```
#### Return 
  ```
  ```

### Generate the training and validation batches of pseudo spots

```python
generate_train_valid_batches(scRNA_file='scRNA.csv', scRNA_label_file='scRNA_label.csv', pseudo_data_path= './batch_data/')
```
#### Description
  ```
  Generate the training and validation batches of pseudo spots.
  ```
#### Parameters  
  ```
  scRNA_data_file       file used to save processed single cell data.
  scRNA_label_file      file used to save annotation information of single cell data.
  pseudo_data_fold      file fold stored the training and validation batches of pseudo spots.  
  ```


### Train stVAE
```python
train_stVAE(spatial_data_file='stRNA.csv', mu_expr_file='mu_gene_expression.csv', disper_file='disp_gene_expression.csv', n_epochs=2000, save_weight=True, load_weight=False)
```
#### Description
  ```
  Train stVAE model.
  ```
#### Parameters  
  ```
  spatial_data_file     file used to save processed spatial transcriptomics.
  mu_expr_file          file used to save cell-type specific mean exrepssion of genes.
  disper_file           file used to save gene dispersion.
  n_epochs              number of epochs to train stVAE.
  save_weight           if True, the weights of stVAE are saved in file 'model_weight.pkl'.
  load_weight           if True, stVAE load model weights from file 'model_weight.pkl'.
  ```
#### Return 
  ```
  the trained model and the list of cell types
  ```

### Train stVAE with pseudo data
```python
train_stVAE_with_pseudo_data(spatial_data_file='stRNA.csv', pseudo_data_fold='./batch_data/', mu_expr_file='mu_gene_expression.csv', disper_file='disp_gene_expression.csv', n_epochs=1000, save_weight=True, load_weight=False)
```
#### Description
  ```
  Train stVAE model with pseudo data.
  ```
#### Parameters  
  ```
  spatial_data_file     file used to save processed spatial transcriptomics.
  pseudo_data_fold      file fold stored the training and validation batches of pseudo spots.  
  mu_expr_file          file used to save cell-type specific mean exrepssion of genes.
  disper_file           file used to save gene dispersion.
  n_epochs              number of epochs to train stVAE.
  save_weight           if True, the weights of stVAE are saved in file 'model_weight.pkl'.
  load_weight           if True, stVAE load model weights from file 'model_weight.pkl'.
  ```
#### Return 
  ```
  the trained model and the list of cell types
  ```

### Get trained stVAE model
```python
get_trained_stVAE(mu_expr_file='mu_gene_expression.csv', weight_file = 'model_weight.pkl')
```
#### Description
  ```
  Get the trained stVAE model.
  ```
#### Parameters  
  ``` 
  mu_expr_file          file saved cell-type specific mean exrepssion of genes.
  weight_file           file saved stVAE model weights.
  ```
#### Return 
  ```
  the trained model and the list of cell types
  ```

### Infer cell type proportions of spots
```python
get_proportions(model, cell_type_list, spatial_data_file='stRNA.csv')
```
#### Description
  ```
  Infer cell type proportions of spots.
  ```
#### Parameters  
  ```
  model                 the trained stVAE model
  cell_type_list        the list of cell types
  spatial_data_file     spatial transcriptomics data file.
  ```
#### Return 
  ```
  Inferred cell type proportions of spots
  ```

## Data
All propcessed data could be downloaded from the shared link: https://drive.google.com/drive/folders/11djR7vxr6Y1VTpz2EVJKH3MvJNGm9VoR?usp=share_link  

## Example workflow
### Import function and datasets
```python
import anndata
from stVAE import get_cell_type_profile
from stVAE import generate_train_valid_batches
from stVAE import train_stVAE, train_stVAE_with_pseudo_data, get_proportions

sc_file = './stVAE data/mouse brain/Stereo-seq/sc_1857.h5ad'
st_file = './stVAE data/mouse brain/Stereo-seq/st_1857.h5ad'
st_adata=anndata.read_h5ad(st_file)
sc_adata=anndata.read_h5ad(sc_file)
```

### Calulate the cell-type specific mean expression level of genes and gene-specific dispersion parameters
```python
get_cell_type_profile(sc_adata, st_adata)
```

### Train stVAE with spatial transcriptomics
```python
model, cell_type_list = train_stVAE()
```

### Train stVAE with pseudo spots and spatial transcriptomics (option)
```python
generate_train_valid_batches()
model, cell_type_list = train_stVAE_with_pseudo_data()
```

### Get inferred cell type proportions
```python
result = get_proportions(model, cell_type_list)
result.to_csv('result.csv')
```
