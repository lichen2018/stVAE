from scvi.external import RNAStereoscope
from scvi import REGISTRY_KEYS
import anndata
from collections import Counter
import pandas as pd
import numpy as np

def get_cell_type_profile(sc_adata, st_adata, mu_expr_file='mu_gene_expression.csv', disper_file='disp_gene_expression.csv', scRNA_data_file='scRNA.csv', scRNA_label_file='scRNA_label.csv', spatial_data_file='stRNA.csv', n_epochs=250):
    """
    Preprocessing data, including calculating cell-type specific mean expression of genes and gene-specific dispersion parameters, 
    generating processed single cell data and annotations used to construct pseudo spots, and generating processed spatial transcriptomic data.

    Parameters
    ----------
    sc_adata
        single cell anndata file.
    st_adata
        spatial transcriptomic anndata file.
    mu_expr_file
        File used to save cell-type specific mean exrepssion of genes
    disper_file
        File used to save gene dispersion
    scRNA_data_file
        File used to save processed single cell data
    scRNA_label_file
        File used to save annotation of single cell data
    spatial_data_file
        File used to save processed spatial transcriptomic data
    n_epochs
        Number of epochs to train for single cell model


    Return
    ----------

    """

    sc_adata = sc_adata.copy()
    RNAStereoscope.setup_anndata(sc_adata, labels_key = "cell_type")
    sc_model = RNAStereoscope(sc_adata)
    sc_model.train(max_epochs = n_epochs)
    sc_model.save("scmodel", overwrite=True)


    count_ct_dict = Counter(list(sc_adata.obs['cell_type']))
    filter_ct = list(count_ct_dict.keys())
    mu_expr = []
    for i in range(len(filter_ct)):
        ct = filter_ct[i]
        ct_idx = list(sc_model.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY).categorical_mapping).index(ct)
        ct_expr = sc_model.module.get_params()[0][:,ct_idx]
        mu_expr.append(ct_expr)

    common_gene_lst = list(sc_adata.var_names)
    pd.DataFrame(data=np.array(mu_expr), columns=common_gene_lst, index=filter_ct).to_csv(mu_expr_file)
    sc_mu_expr = pd.DataFrame(data=np.array(mu_expr), columns=common_gene_lst, index=filter_ct)
    
    import csv
    with open(disper_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(sc_model.module.get_params()[1])
        f.close()



    #pd.DataFrame(data=sc_adata[:,common_gene_lst].X.A, columns=common_gene_lst, index=sc_adata.obs_names).to_csv(scRNA_data_file)
    #pd.DataFrame(data=st_adata[:,common_gene_lst].X.A, columns=common_gene_lst, index=st_adata.obs_names ).to_csv(spatial_data_file)
    #scRNA_data = pd.DataFrame(data=sc_adata[:,common_gene_lst].X.A, columns=common_gene_lst, index=sc_adata.obs_names)
    #scRNA_label = pd.DataFrame.from_dict(label_dict)
    #stRNA_data = pd.DataFrame(data=st_adata[:,common_gene_lst].X.A, columns=common_gene_lst, index=st_adata.obs_names )

