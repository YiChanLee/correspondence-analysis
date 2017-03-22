"""
Test 1 of correspondence analysis
Alan J Izenman - Modern Multivariate Statistical Techniques
hair-eye exmple
"""

import ca
import pandas as pd
import seaborn as sns


transfer = ca.CA()

hair_eye = pd.read_csv('./datasets/hair_eye.csv', index_col=0)
print(hair_eye.head())

transfer.fit(hair_eye)
print('centered correspondence matrix:')
print(transfer.centr_corrspnd_mat_)

print('Pearson residual:')
print(transfer.pearson_resd_)

pcs_row, pcs_col = \
    transfer.get_princpl_coords_df(row_categories=hair_eye.index,
                                   col_categories=hair_eye.columns)
print('Principal coordinates of row variables in DataFrame:')
print(pcs_row)
print(pcs_col)


fig, ax = sns.plt.subplots()
sns.regplot('Dim 0', 'Dim 1', data=pcs_row, fit_reg=False, ax=ax)
sns.regplot('Dim 0', 'Dim 1', data=pcs_col, fit_reg=False, ax=ax)
for i, txt in enumerate(list(hair_eye.index)):
    ax.annotate(txt + ' eyes',
                (pcs_row.iloc[i]['Dim 0'], pcs_row.iloc[i]['Dim 1']))
for i, txt in enumerate(list(hair_eye.columns)):
    ax.annotate(txt + ' hair',
                (pcs_col.iloc[i]['Dim 0'], pcs_col.iloc[i]['Dim 1']))
ax.set_xlabel('Dim 0')
ax.set_ylabel('Dim 1')
sns.plt.show()
