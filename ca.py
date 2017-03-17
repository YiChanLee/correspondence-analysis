import numpy as np
import pandas as pd


class CA:

    def __init__(self, n_components=None, copy=True):
        self.n_components = n_components
        self.copy = copy
        self.avg_col_profile_ = None

    def fit(self, contingency_table):
        grand_total = contingency_table.sum().sum()

        row_means = contingency_table.sum(axis=1) / grand_total
        col_means = contingency_table.sum(axis=0) / grand_total

        corrspnd_mat = contingency_table / grand_total
        expc_freq = np.kron(row_means.reshape(-1, 1), col_means.reshape(1, -1))
        centr_corrspnd_mat = corrspnd_mat - expc_freq

        chi_squared = \
            grand_total * ((centr_corrspnd_mat ** 2) / expc_freq).sum().sum()

        Dr_sqrt_inv = np.diag(1 / np.sqrt(row_means))
        Dc_sqrt_inv = np.diag(1 / np.sqrt(col_means))

        pearson_resd = Dr_sqrt_inv @ centr_corrspnd_mat @ Dc_sqrt_inv

        principal_inertias, _ = np.linalg.eigh(pearson_resd.T @ pearson_resd)

        U, D_lamb, V_T = np.linalg.svd(pearson_resd)

        min_size = np.min(contingency_table.shape)
        D_lamb_mat = np.zeros(contingency_table.shape)
        D_lamb_mat[:min_size, :min_size] = np.diag(D_lamb)

        pcs_row_profiles = Dr_sqrt_inv @ U @  D_lamb_mat
        pcs_col_profiles = Dc_sqrt_inv @ V_T.T @ D_lamb_mat.T

        if self.n_components is None:
            self.n_components = (D_lamb > 1e-16).sum()

        self.grand_total_ = grand_total
        self.corrspnd_mat_ = corrspnd_mat
        self.chi_squared_ = chi_squared
        self.pearson_resd_ = pearson_resd
        self.principal_inertias_ = principal_inertias
        self.pcs_row = pcs_row_profiles[:, :self.n_components].T
        self.pcs_col = pcs_col_profiles[:, :self.n_components].T

    def get_pcs_df(self, row_categories=None, col_categories=None):
        row_pcs_df = pd.DataFrame(data=self.pcs_row, columns=row_categories)
        col_pcs_df = pd.DataFrame(data=self.pcs_col, columns=col_categories)
        return row_pcs_df, col_pcs_df
