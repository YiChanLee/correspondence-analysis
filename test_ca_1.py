"""
Test 2 of correspondence analysis
Correspondence Analysis in R: The Ultimate Guide for the Analysis,
the Visualization and the Interpretation - R software and data mining
http://www.sthda.com/english/wiki/correspondence-analysis-in-r-the-ultimate-guide-for-the-analysis-the-visualization-and-the-interpretation-r-software-and-data-mining#ca-scatter-plot-biplot-of-row-and-column-variables
The data is a contingency table containing 13 housetasks and their repartition
in the couple
"""

import ca
import pandas as pd
import numpy as np
import seaborn as sns

transfer = ca.CA()

house_tasks = pd.read_csv('./datasets/house_tasks.csv', index_col=0)

transfer.fit(house_tasks)

pcs_row, pcs_col = \
    transfer.get_princpl_coords_df(row_categories=house_tasks.index,
                                   col_categories=house_tasks.columns)
pcs_row['Dim 1'] = -pcs_row['Dim 1']
pcs_col['Dim 1'] = -pcs_col['Dim 1']
print('Principal coordinates of row variables in DataFrame:')
print(pcs_row)
print(pcs_col)

variances = transfer.principal_inertias_
percent_explnd_var = (variances / variances.sum()) * 100

fig, ax = sns.plt.subplots()
sns.barplot(x=np.arange(1, 5), y=percent_explnd_var[:4], ax=ax)
ax.set_xlabel('Dimensions')
ax.set_ylabel('Percentage of explained variances')
var_text = ['{:.1f}%'.format(pers) for pers in percent_explnd_var[:4]]
for i, txt in enumerate(var_text):
    ax.annotate(txt, (i, percent_explnd_var[i]),
                horizontalalignment='center',
                verticalalignment='center')
sns.plt.show()

fig, ax = sns.plt.subplots()
sns.regplot('Dim 0', 'Dim 1', data=pcs_row, fit_reg=False, ax=ax)
sns.regplot('Dim 0', 'Dim 1', data=pcs_col, fit_reg=False, ax=ax)
for i, txt in enumerate(list(house_tasks.index)):
    ax.annotate(txt, (pcs_row.iloc[i]['Dim 0'], pcs_row.iloc[i]['Dim 1']))
for i, txt in enumerate(list(house_tasks.columns)):
    ax.annotate(txt, (pcs_col.iloc[i]['Dim 0'], pcs_col.iloc[i]['Dim 1']))
ax.set_xlabel('Dim 0')
ax.set_ylabel('Dim 1')
sns.plt.show()
