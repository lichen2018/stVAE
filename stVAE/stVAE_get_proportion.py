import torch
import warnings
from torch import nn
from typing import Optional, Tuple, Union
import torch.nn.functional as F
from torch.distributions import Distribution, Gamma, Poisson, constraints
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)
import torch.optim as optim
import os
import pandas as pd
import numpy as np
import torch.nn.functional as f
#from torchsummary import summary
from torch.autograd import Variable
from torch.distributions import NegativeBinomial
from sparsemax import Sparsemax
import os
torch.manual_seed(0)



class ST_Vae(nn.Module):
    def __init__(self, n_input, n_class, n_layers, n_latent, n_hidden = 1024, dropout_rate = 0.1, use_batch_norm = True, use_layer_norm= True, use_activation= True):
        super(ST_Vae, self).__init__()
        layers_dim = [n_input] + (n_layers - 1) * [n_hidden]
        modules = []
        for i, (n_in, n_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:])):
            modules.append(
                nn.Sequential(
                    nn.Linear(n_in, n_out),
                    nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001) if use_batch_norm else None,
                    nn.LayerNorm(n_out, elementwise_affine=False) if use_layer_norm else None,
                    nn.LeakyReLU() if use_activation else None,
                    nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,)
            )
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(n_hidden, n_latent)
        self.fc_var = nn.Linear(n_hidden, n_latent)
        layers_dim = [n_latent] + (n_layers - 1) * [n_hidden]
        modules = []
        for i, (n_in, n_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:])):
            modules.append(
                nn.Sequential(
                    nn.Linear(n_in, n_out),
                    nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001) if use_batch_norm else None,
                    nn.LayerNorm(n_out, elementwise_affine=False) if use_layer_norm else None,
                    nn.LeakyReLU() if use_activation else None,
                    nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,)
            )
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                    nn.Linear(layers_dim[-1], n_class),
                    nn.BatchNorm1d(n_class, momentum=0.01, eps=0.001) if use_batch_norm else None,
                    nn.LayerNorm(n_class, elementwise_affine=False) if use_layer_norm else None,
                    nn.Tanh())
        self.px_r =nn.Linear(layers_dim[-1], n_input)
        self.scale =nn.Parameter(torch.randn(n_input))
        self.additive =nn.Parameter(torch.randn(n_input))

    def encode(self, input_x):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x F]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input_x)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x F]
        """
        result = self.decoder(z)
        y_proportion = self.final_layer(result)
        p_r = torch.exp(self.px_r(result))
        return y_proportion, p_r
    def forward(self, xs):
        mu, log_var = self.encode(xs)
        z = self.reparameterize(mu, log_var)
        y_proportion, p_r = self.decode(z)
        scale = nn.Sigmoid()(self.scale)
        additive = torch.exp(self.additive)

        return  y_proportion, scale, additive, mu, log_var


def loss_function(pred_ys, ys, xs, mu, log_var, umi_counts, mu_expr, px_r, scale, additive):
    """
    Computes the VAE loss function.
    KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
    :param args:
    :param kwargs:
    :return:
    """
    likelihood_weight = 0.0001

    kld_weight = 0.0000025

    library_size = xs.sum(axis=1)


    library_size_comp = torch.mul(pred_ys.T, library_size).T
    px_rate = torch.matmul(library_size_comp, nn.Softplus()(mu_expr))
    #px_rate = torch.add(torch.mul(px_rate, scale), additive)
    likelihood_loss = torch.mean(-NegativeBinomial(px_rate, logits=px_r).log_prob(umi_counts).sum(dim=-1))
    prop_loss = nn.L1Loss()(pred_ys, ys)
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    loss = likelihood_weight*likelihood_loss + 10*prop_loss + kld_weight * kld_loss
    return {'loss': loss, 'likelihood_loss':likelihood_loss.detach(), 'prop_loss':prop_loss.detach(), 'KLD':-kld_loss.detach()}


def loss_function_st(pred_ys, xs, mu, log_var, umi_counts, mu_expr, px_r, scale, additive):
    """
    Computes the VAE loss function.
    KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
    :param args:
    :param kwargs:
    :return:
    """
    likelihood_weight = 0.005

    kld_weight = 0.0000025

    library_size = xs.sum(axis=1)


    library_size_comp = torch.mul(pred_ys.T, library_size).T
    px_rate = torch.matmul(library_size_comp, nn.Softplus()(mu_expr))
    px_rate = torch.add(torch.mul(px_rate, scale), additive)
    likelihood_loss = torch.mean(-NegativeBinomial(px_rate, logits=px_r).log_prob(umi_counts).sum(dim=-1))
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    loss = likelihood_weight*likelihood_loss + kld_weight * kld_loss
    return {'loss': loss, 'likelihood_loss':likelihood_loss.detach(), 'KLD':-kld_loss.detach()}


def get_trained_stVAE(mu_expr_file='mu_gene_expression.csv', weight_file = 'model_weight.pkl'):
    mu_expr_df = pd.read_csv(mu_expr_file, delimiter = ',', header = 0, index_col = 0)
    cell_type_list = list(mu_expr_df.index)
    mu_expr = mu_expr_df.values.astype(np.float32)
    n_class = mu_expr.shape[0]
    feature_num = mu_expr.shape[1]

    model = ST_Vae(feature_num, n_class, n_layers = 3, n_latent = 128)
    model.load_state_dict(torch.load(weight_file))
    return model, cell_type_list 


def train_stVAE(spatial_data_file='stRNA.csv', mu_expr_file='mu_gene_expression.csv', disper_file='disp_gene_expression.csv', n_epochs=2000, save_weight=True, load_weight=False):
    """
    Train stVAE model.
    Parameters
    ----------
    spatial_data_file
        spatial transcriptomics data file.
    mu_expr_file
        File of cell-type specific mean exrepssion of genes
    disper_file
        File of gene dispersion
    n_epochs
        Number of total epochs in training.
    save_weight
        If True, the weights of stVAE are saved in file 'model_weight.pkl'
    load_weight
        If True, stVAE load model weights from file 'model_weight.pkl'

    Return
    ----------
        trained model and cell type list   
    """


    st_data_df = pd.read_csv(spatial_data_file, delimiter=',', header=0, index_col=0)
    st_data = st_data_df.values.astype(np.float32)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    mu_expr_df = pd.read_csv(mu_expr_file, delimiter = ',', header = 0, index_col = 0)
    cell_type_list = list(mu_expr_df.index)



    mu_expr = torch.tensor(pd.read_csv(mu_expr_file, delimiter = ',', header = 0, index_col = 0).values.astype(np.float32))
    disper = torch.tensor(pd.read_csv(disper_file, delimiter = ',', header = None).values.astype(np.float32))[0]


    n_class = mu_expr.shape[0]
    feature_num = mu_expr.shape[1]

    model = ST_Vae(feature_num, n_class, n_layers = 3, n_latent = 128)
    print(model)

    use_cuda = True
    if use_cuda:
        model.cuda()
        mu_expr = mu_expr.to(device)

    train_result_log = 'train_result_log.txt'
    
    if load_weight is True:
        weights_file = 'model_weight.pkl'
        if os.path.isfile(weights_file) is False:
            print('Model weights file does not exist! The program will interrupt, please set load_weight flag to False!')
            exit(0)
        model.load_state_dict(torch.load(weights_file))

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        likelihood_loss = 0.0
        prop_loss = 0.0
        kld_loss = 0.0
        st_running_loss = 0.0
        st_likelihood_loss = 0.0
        st_kld_loss = 0.0
        for i in range(int(st_data.shape[0]/120)):
            if i < int(st_data.shape[0]/120):
                jdx = i*120
                data_arr = st_data[jdx:jdx+120,:]
                umi_counts = torch.tensor(data_arr.astype(np.float32))
                umi_counts = umi_counts[umi_counts.sum(dim=1) != 0]
                if umi_counts.nelement() == 0:
                    continue
                xs = umi_counts
                batch_size_tmp = xs.shape[0]
                sc_px_r = disper.repeat(batch_size_tmp, 1)
                if use_cuda:
                    xs = xs.to(device)
                    umi_counts = umi_counts.to(device)
                    sc_px_r = sc_px_r.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                pred_ys, scale, additive, mu, log_var = model(xs)
                pred_ys = Sparsemax()(pred_ys)
                #sc_rate: tk->itk; st_rate: ik->itk
                scale = scale.repeat(batch_size_tmp, 1)
                additive= additive.repeat(batch_size_tmp, 1)
                loss = loss_function_st(pred_ys, xs, mu, log_var, umi_counts, mu_expr, sc_px_r, scale, additive)
                loss['loss'].backward()
                optimizer.step()
                # print statistics
                st_running_loss += loss['loss'].item()
                st_likelihood_loss += loss['likelihood_loss'].item()
                st_kld_loss += loss['KLD'].item()


        if epoch == 0:
            fo = open(train_result_log, 'w')
        else:
            fo = open(train_result_log, 'a')
        fo.write("[epoch %03d] st_likelihood_loss: %.4f  st_kld_loss: %.4f\n" % (epoch, st_likelihood_loss/int(st_data.shape[0]/120), st_kld_loss/int(st_data.shape[0]/120)))
        fo.close()

    if save_weight:
        save_path = 'model_weight.pkl'
        torch.save(model.state_dict(),save_path)


    return model, cell_type_list






def train_stVAE_with_pseudo_data(spatial_data_file='stRNA.csv', pseudo_data_fold='./batch_data/', mu_expr_file='mu_gene_expression.csv', disper_file='disp_gene_expression.csv', n_epochs=1000, save_weight=True, load_weight=False):
    """
    Train stVAE model with pseudo data.
    Parameters
    ----------
    spatial_data_file
        spatial transcriptomics data file.
    pseudo_data_fold
        pseudo data file.
    mu_expr_file
        File of cell-type specific mean exrepssion of genes
    disper_file
        File of gene dispersion
    n_epochs
        Number of total epochs in training.
    save_weight
        If True, stVAE save model weights to file 'model_weight.pkl'
    load_weight
        If True, stVAE load model weights from file 'model_weight.pkl'

    Return
    ----------
        trained model and cell type list   
    """

    train_data_file = []
    validate_data_file = []
    train_label_file = []
    validate_label_file = []

    for file_name in os.listdir(pseudo_data_fold):
        if file_name.split('_')[0] == 'train' and file_name.split('_')[1] == 'data':
            train_data_file.append(np.genfromtxt(pseudo_data_fold+file_name, delimiter=','))
            label_file_name = file_name.replace('data','label')

            train_label_file.append(np.genfromtxt(pseudo_data_fold+label_file_name, delimiter=','))

        if file_name.split('_')[0] == 'validate' and file_name.split('_')[1] == 'data':
            validate_data_file.append(np.genfromtxt(pseudo_data_fold+file_name, delimiter=','))
            label_file_name = file_name.replace('data','label')
            validate_label_file.append(np.genfromtxt(pseudo_data_fold+label_file_name, delimiter=','))


    st_data_df = pd.read_csv(spatial_data_file, delimiter=',', header=0, index_col=0)
    st_data = st_data_df.values.astype(np.float32)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    mu_expr_df = pd.read_csv(mu_expr_file, delimiter = ',', header = 0, index_col = 0)
    cell_type_list = list(mu_expr_df.index)


    mu_expr = torch.tensor(pd.read_csv(mu_expr_file, delimiter = ',', header = 0, index_col = 0).values.astype(np.float32))
    disper = torch.tensor(pd.read_csv(disper_file, delimiter = ',', header = None).values.astype(np.float32))[0]


    n_class = mu_expr.shape[0]
    feature_num = mu_expr.shape[1]

    model = ST_Vae(feature_num, n_class, n_layers = 3, n_latent = 128)
    print(model)

    TEST_FREQUENCY = 50
    use_cuda = True
    if use_cuda:
        model.cuda()
        mu_expr = mu_expr.to(device)

    train_result_log = 'train_result_log.txt'
    valid_result_log = 'valid_result_log.txt'
    if load_weight is True:
        weights_file = 'model_weight.pkl'
        if os.path.isfile(weights_file) is False:
            print('Model weights file does not exist! The program will interrupt, please set load_weight flag to False!')
            exit(0)
        model.load_state_dict(torch.load(weights_file))

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        likelihood_loss = 0.0
        prop_loss = 0.0
        kld_loss = 0.0
        st_running_loss = 0.0
        st_likelihood_loss = 0.0
        st_kld_loss = 0.0
        for i in range(max(int(st_data.shape[0]/120), 3000)):
            idx = i
            if i >= 3000:
                idx = int(i%3000)
            data_arr = train_data_file[idx]
            data_arr = data_arr
            data_arr = data_arr.astype(int)
            label_arr = train_label_file[idx]
            umi_counts = torch.tensor(data_arr.astype(np.float32))
            ys = torch.tensor(label_arr.astype(np.float32))
            ys = ys[umi_counts.sum(dim=1) != 0]
            umi_counts = umi_counts[umi_counts.sum(dim=1) != 0]
            if umi_counts.nelement() == 0:
                continue
            xs = umi_counts
            batch_size_tmp = xs.shape[0]
            sc_px_r = disper.repeat(batch_size_tmp, 1)
            if use_cuda:
                xs = xs.to(device)
                umi_counts = umi_counts.to(device)
                ys = ys.to(device)
                sc_px_r = sc_px_r.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            pred_ys, scale, additive, mu, log_var = model(xs)
            pred_ys = Sparsemax()(pred_ys)
            #sc_rate: tk->itk; st_rate: ik->itk
            scale = scale.repeat(batch_size_tmp, 1)
            additive= additive.repeat(batch_size_tmp, 1)
            loss = loss_function(pred_ys, ys, xs, mu, log_var, umi_counts, mu_expr, sc_px_r, scale, additive)
            loss['loss'].backward()
            optimizer.step()

            # print statistics
            running_loss += loss['loss'].item()
            likelihood_loss += loss['likelihood_loss'].item()
            prop_loss += loss['prop_loss'].item()
            kld_loss += loss['KLD'].item()

            if i < int(st_data.shape[0]/120):
                jdx = i*120
                data_arr = st_data[jdx:jdx+120,:]
                umi_counts = torch.tensor(data_arr.astype(np.float32))
                umi_counts = umi_counts[umi_counts.sum(dim=1) != 0]
                if umi_counts.nelement() == 0:
                    continue
                xs = umi_counts
                batch_size_tmp = xs.shape[0]
                sc_px_r = disper.repeat(batch_size_tmp, 1)
                if use_cuda:
                    xs = xs.to(device)
                    umi_counts = umi_counts.to(device)
                    sc_px_r = sc_px_r.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                pred_ys, scale, additive, mu, log_var = model(xs)
                pred_ys = Sparsemax()(pred_ys)
                #sc_rate: tk->itk; st_rate: ik->itk
                scale = scale.repeat(batch_size_tmp, 1)
                additive= additive.repeat(batch_size_tmp, 1)

                loss = loss_function_st(pred_ys, xs, mu, log_var, umi_counts, mu_expr, sc_px_r, scale, additive)
                loss['loss'].backward()
                optimizer.step()
                # print statistics
                st_running_loss += loss['loss'].item()
                st_likelihood_loss += loss['likelihood_loss'].item()
                st_kld_loss += loss['KLD'].item()


        if epoch == 0:
            fo = open(train_result_log, 'w')
        else:
            fo = open(train_result_log, 'a')
        fo.write("[epoch %03d] loss: %.4f  likelihood loss: %.4f   proportion loss: %.4f  kld loss: %.4f  st_likelihood_loss: %.4f  st_kld_loss: %.4f\n" % (epoch, running_loss/max(int(st_data.shape[0]/120), 3000), likelihood_loss/max(int(st_data.shape[0]/120), 3000), prop_loss/max(int(st_data.shape[0]/120), 3000), kld_loss/max(int(st_data.shape[0]/120), 3000), st_likelihood_loss/int(st_data.shape[0]/120), st_kld_loss/int(st_data.shape[0]/120)))
        fo.close()
        if epoch % TEST_FREQUENCY == 0:
            running_loss = 0.0
            likelihood_loss = 0.0
            prop_loss = 0.0
            kld_loss = 0.0
            with torch.no_grad():
                for i in range(min(len(validate_data_file),600)):
                    data_arr = validate_data_file[i]
                    data_arr = data_arr.astype(int)
                    label_arr = validate_label_file[i]
                    umi_counts = torch.tensor(data_arr.astype(np.float32))
                    ys = torch.tensor(label_arr.astype(np.float32))
                    ys = ys[umi_counts.sum(dim=1) != 0]
                    umi_counts = umi_counts[umi_counts.sum(dim=1) != 0]
                    if umi_counts.nelement() == 0:
                        continue
                    xs = umi_counts
                    batch_size_tmp = xs.shape[0]
                    sc_px_r = disper.repeat(batch_size_tmp, 1)
                    if use_cuda:
                        xs = xs.to(device)
                        umi_counts = umi_counts.to(device)
                        ys = ys.to(device)
                        sc_px_r = sc_px_r.to(device)
                    pred_ys, scale, additive, mu, log_var = model(xs)
                    pred_ys = Sparsemax()(pred_ys)
                    scale = scale.repeat(batch_size_tmp, 1)
                    additive= additive.repeat(batch_size_tmp, 1)
                    loss = loss_function(pred_ys, ys, xs, mu, log_var, umi_counts, mu_expr, sc_px_r, scale, additive)
                    running_loss += loss['loss'].item()
                    likelihood_loss += loss['likelihood_loss'].item()
                    prop_loss += loss['prop_loss'].item()
                    kld_loss += loss['KLD'].item()
            if epoch == 0:
                fo = open(valid_result_log, 'w')
            else:
                fo = open(valid_result_log, 'a')
            fo.write("[epoch %03d] loss: %.4f  likelihood loss: %.4f   proportion loss: %.4f  kld loss: %.4f\n" % (epoch, running_loss/min(len(validate_data_file),600), likelihood_loss/min(len(validate_data_file),600), prop_loss/min(len(validate_data_file),600), kld_loss/min(len(validate_data_file),600)))
            fo.close()
    if save_weight:
        save_path = 'model_weight.pkl'
        torch.save(model.state_dict(),save_path)
 
    return model, cell_type_list



def get_proportions(model, cell_type_list, spatial_data_file='stRNA.csv'):
    """
    Infer cell type proportions of spots.
    Parameters
    ----------
    model
        trained stVAE model
    cell_type_list
        a list of cell types
    spatial_data_file
        spatial transcriptomics data file.

    Return
    ----------
        Inferred cell type proportions of spots   
    """
    model.eval()

    st_data_df = pd.read_csv(spatial_data_file, delimiter=',', header=0, index_col=0)
    st_data = st_data_df.values.astype(np.float32)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    use_cuda = True
    if use_cuda:
        model.cuda()

    result = []
    with torch.no_grad():
        for i in range(0, st_data.shape[0], 120):
            data_arr = st_data[i:i+120,:]
            umi_counts = torch.tensor(data_arr)
            xs = umi_counts
            if use_cuda:
                xs = xs.to(device)
            pred_ys, scale, additive, mu, log_var = model(xs)
            pred_ys_norm = Sparsemax()(pred_ys)
            tmp_pred_ys = np.array(pred_ys_norm.cpu())
            if i == 0:
                result = tmp_pred_ys
            else:
                result = np.concatenate((result, tmp_pred_ys), axis=0)
    result = pd.DataFrame(data=result, columns=cell_type_list, index=list(st_data_df.index)) 
    return result
