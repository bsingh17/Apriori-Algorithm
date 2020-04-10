import numpy as np
import pandas as pd


dataset=pd.read_csv('store_data.csv')

data=[]

for i in range(0,7501):
    data.append([str(dataset.values[i,j]) for j in range (0,20) ])
    
    
from apyori import apriori
rules=apriori(data,min_support=0.0045,min_confidence=0.2,min_lift=3,min_length=2)
rules=list(rules)

print(rules[0])