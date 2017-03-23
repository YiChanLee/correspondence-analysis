"""
Test of correspondence analysis for supplemantary points
Correspondence Analysis in R: The Ultimate Guide for the Analysis,
the Visualization and the Interpretation - R software and data mining
http://www.sthda.com/english/wiki/correspondence-analysis-in-r-the-ultimate-guide-for-the-analysis-the-visualization-and-the-interpretation-r-software-and-data-mining#correspondence-analysis-using-supplementary-rows-and-columns
The data used here is a contingency table describing the answers given by
different categories of people to the following question: What are the reasons
that can make hesitate a woman or a couple to have children?
"""
import ca
import pandas as pd
import seaborn as sns


transfrmr = ca.CA()
children = pd.read_csv('./datasets/children.csv', index_col=0)
X = children.loc[:'work', :'university']

transfrmr.fit(X)

pcs_row, pcs_col = \
    transfrmr.get_princpl_coords_df(row_categories=X.index,
                                    col_categories=X.columns)
pcs_row['Dim 1'] = -pcs_row['Dim 1']
pcs_col['Dim 1'] = -pcs_col['Dim 1']
print('Principal coordinates of row variables in DataFrame:')
print(pcs_row)
print(pcs_col)

supp_rows = children.loc['comfort':, :'university']
supp_cols = children.loc[:'work', 'thirty':]
new_supp_rows = transfrmr.transform(supp_rows)
new_supp_cols = transfrmr.transform(supp_cols, row=False)
new_supp_rows[:, 1] = -new_supp_rows[:, 1]
new_supp_cols[:, 1] = -new_supp_cols[:, 1]
print(new_supp_rows)
print(new_supp_cols)

fig, ax = sns.plt.subplots()
sns.regplot('Dim 0', 'Dim 1', data=pcs_row, ax=ax, fit_reg=False, label='rows')
sns.regplot('Dim 0', 'Dim 1', data=pcs_col, ax=ax, fit_reg=False, label='cols')
sns.regplot(new_supp_rows[:, 0], new_supp_rows[:, 1],
            ax=ax, fit_reg=False, label='supp rows')
sns.regplot(new_supp_cols[:, 0], new_supp_cols[:, 1],
            ax=ax, fit_reg=False, label='supp cols')
for i, txt in enumerate(list(X.index)):
    ax.annotate(txt, (pcs_row.iloc[i]['Dim 0'], pcs_row.iloc[i]['Dim 1']))
for i, txt in enumerate(list(X.columns)):
    ax.annotate(txt, (pcs_col.iloc[i]['Dim 0'], pcs_col.iloc[i]['Dim 1']))
for i, txt in enumerate(list(supp_rows.index)):
    ax.annotate(txt, (new_supp_rows[i, 0], new_supp_rows[i, 1]))
for i, txt in enumerate(list(supp_cols.columns)):
    ax.annotate(txt, (new_supp_cols[i, 0], new_supp_cols[i, 1]))
ax.legend()
sns.plt.show()
