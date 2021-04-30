import pandas as pd
pd.set_option('display.max_columns',None)
pd.set_option('display.expand_frame_repr',False)
from mlxtend.frequent_patterns import apriori, association_rules

df_ = pd.read_excel("dataset/online_retail_II.xlsx",sheet_name="Year 2010-2011")
df=df_.copy()
df.info()
df.head()

from dataset.helpers import crm_data_prep

check_df(df)

df=crm_data_prep(df)
check_df(df)

df_g=df[(df["Country"]=="Germany")&(df["StockCode"]!="POST")]
check_df(df_g)

def create_invoice_product_df(dataframe):
    return dataframe.groupby(["Invoice","StockCode"])["Quantity"].sum().unstack().fillna(0).\
    applymap(lambda x:1 if x>0 else 0)

g_inv_pro_df=create_invoice_product_df(df_g)
g_inv_pro_df.head()

frequent_itemsets=apriori(g_inv_pro_df,min_support=0.01,use_colnames=True)
frequent_itemsets.sort_values("support",ascending=False)

rules=association_rules(frequent_itemsets,metric="support",min_threshold=0.01)
rules.head()
rules.sort_values("lift",ascending=False).head()