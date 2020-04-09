import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
dataset=pd.read_excel('Online Retail.xlsx')

#removing the extra space in the description column
dataset['Description']=dataset['Description'].str.strip()

#dropping the rows without invoice number
dataset.dropna(axis=0,subset=['InvoiceNo'],inplace=True)
dataset['InvoiceNo']=dataset['InvoiceNo'].astype(str)

#removing all the paymets done by credit card
dataset=dataset[~dataset['InvoiceNo'].str.contains('C')]

#splitting the whole data

basket_France=(dataset[dataset['Country'] =="France"] 
          .groupby(['InvoiceNo', 'Description'])['Quantity'] 
          .sum().unstack().reset_index().fillna(0) 
          .set_index('InvoiceNo')) 
basket_Uk=(dataset[dataset['Country']=='United Kingdom']
          .groupby(['InvoiceNo','Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

#Performing hot encoding
def hot_encode(x):
    if x<=0:
        return 0
    if x>=1:
        return 1
    
basket_France=basket_France.applymap(hot_encode)
basket_Uk=basket_Uk.applymap(hot_encode)

#building the models
fr_items=apriori(basket_France,min_support=0.05,use_colnames=True)
rules=association_rules(fr_items,metric='lift',min_threshold=1)
rules=rules.sort_values(['confidence','lift'],ascending=[False,False])
print(rules.head())
fr2_items=apriori(basket_Uk,min_support=0.01,use_colnames=True)
rules1=association_rules(fr2_items,metric='lift',min_threshold=1)
rules1=rules1.sort_values(['confidence','lift'],ascending=[False,False])