"""
Test of multiple correspondence analysis
Comparing to FactoMineR package in R
http://www.sthda.com/english/wiki/multiple-correspondence-analysis-essentials-interpretation-and-application-to-investigate-the-associations-between-categories-of-multiple-qualitative-variables-r-software-and-data-mining
This data is a result from a survey carried out on children of primary school
who suffered from food poisoning. They were asked about their symptoms and
about what they ate.
"""

import ca
import pandas as pd
import numpy as np
import seaborn as sns

transfer = ca.CA()
poison = pd.read_csv('./datasets/poison.csv')
poison.drop(labels='Age Time Sick Sex'.split(' '), inplace=True, axis=1)
for col in poison.columns:
    poison[col] = poison[col].str.split('_').str.get(1)
poison = pd.get_dummies(poison)
print(poison.head())

transfer.fit(poison)

pcs_row, pcs_col = \
    transfer.get_princpl_coords_df(row_categories=poison.index,
                                   col_categories=poison.columns)
pcs_col['Dim 1'] = -pcs_col['Dim 1']
pcs_row['Dim 1'] = -pcs_row['Dim 1']
print('='*20)
print('Principal coordinates of row variables in DataFrame:')
print(pcs_row[['Dim 0', 'Dim 1']].head(10))
print(pcs_col[['Dim 0', 'Dim 1']])

variances = transfer.principal_inertias_
percent_explnd_var = (variances / variances.sum()) * 100

fig, ax = sns.plt.subplots()
sns.barplot(x=np.arange(1, 11), y=percent_explnd_var[:10], ax=ax)
ax.set_xlabel('Dimensions')
ax.set_ylabel('Percentage of explained variances')
var_text = ['{:.1f}%'.format(pers) for pers in percent_explnd_var[:10]]
for i, txt in enumerate(var_text):
    ax.annotate(txt, (i, percent_explnd_var[i]),
                horizontalalignment='center',
                verticalalignment='center')
sns.plt.show()

fig, ax = sns.plt.subplots()
sns.regplot(x='Dim 0', y='Dim 1', data=pcs_row, fit_reg=False, ax=ax)
for i, txt in enumerate(list(poison.index)):
    ax.annotate(txt, (pcs_row.iloc[i]['Dim 0'], pcs_row.iloc[i]['Dim 1']))
sns.plt.show()

fig, ax = sns.plt.subplots()
sns.regplot(x='Dim 0', y='Dim 1', data=pcs_col, fit_reg=False, ax=ax)
for i, txt in enumerate(list(poison.columns)):
    ax.annotate(txt, (pcs_col.iloc[i]['Dim 0'], pcs_col.iloc[i]['Dim 1']))
sns.plt.show()
