import numpy as np
import pandas as pd


class CA:

    def __init__(self, n_components=None, copy=True):
        self.n_components = n_components

    def fit(self, contingency_table):
        grand_total = contingency_table.sum().sum()

        row_masses = contingency_table.sum(axis=1) / grand_total
        col_masses = contingency_table.sum(axis=0) / grand_total

        corrspnd_mat = contingency_table / grand_total
        expc_freq = np.kron(row_masses.reshape(-1, 1),
                            col_masses.reshape(1, -1))
        centr_corrspnd_mat = corrspnd_mat - expc_freq

        chi_squared = \
            grand_total * ((centr_corrspnd_mat ** 2) / expc_freq).sum().sum()

        Dr_sqrt_inv = np.diag(1 / np.sqrt(row_masses))
        Dc_sqrt_inv = np.diag(1 / np.sqrt(col_masses))

        pearson_resd = Dr_sqrt_inv @ centr_corrspnd_mat @ Dc_sqrt_inv

        principal_inertias, _ = np.linalg.eigh(pearson_resd.T @ pearson_resd)

        U, D_lamb, V_T = np.linalg.svd(pearson_resd)

        min_size = np.min(contingency_table.shape)
        D_lamb_mat = np.zeros(contingency_table.shape)
        D_lamb_mat[:min_size, :min_size] = np.diag(D_lamb)

        princpl_coords_row = Dr_sqrt_inv @ U @  D_lamb_mat
        princpl_coords_col = Dc_sqrt_inv @ V_T.T @ D_lamb_mat.T

        std_coords_row = Dr_sqrt_inv @ U
        std_coords_col = Dc_sqrt_inv @ V_T.T

        if self.n_components is None:
            self.n_components = (D_lamb > 1e-16).sum()

        self.grand_total_ = grand_total
        self.row_masses_ = row_masses
        self.col_masses_ = col_masses
        self.corrspnd_mat_ = corrspnd_mat
        self.centr_corrspnd_mat_ = centr_corrspnd_mat
        self.chi_squared_ = chi_squared
        self.pearson_resd_ = pearson_resd
        self.principal_inertias_ = principal_inertias
        self.princpl_coords_row_ = princpl_coords_row[:, :self.n_components]
        self.princpl_coords_col_ = princpl_coords_col[:, :self.n_components]
        self.std_coords_row_ = std_coords_row
        self.std_coords_col_ = std_coords_col

    def get_princpl_coords_df(self, row_categories=None, col_categories=None):
        inds = ['Dim {}'.format(i) for i in range(self.n_components)]
        row_pcs_df = pd.DataFrame(data=self.princpl_coords_row_,
                                  columns=inds,
                                  index=row_categories)
        col_pcs_df = pd.DataFrame(data=self.princpl_coords_col_,
                                  columns=inds,
                                  index=col_categories)
        return row_pcs_df, col_pcs_df

    def get_std_coords_df(self, row_categories=None, col_categories=None):
        inds_row = \
            ['Dim {}'.format(i) for i in range(self.corrspnd_mat_.shape[0])]
        inds_col = \
            ['Dim {}'.format(i) for i in range(self.corrspnd_mat_.shape[1])]
        row_stds_df = pd.DataFrame(data=self.std_coords_row_,
                                   columns=inds_row,
                                   index=row_categories)
        col_stds_df = pd.DataFrame(data=self.std_coords_col_,
                                   columns=inds_col,
                                   index=col_categories)
        return row_stds_df, col_stds_df
