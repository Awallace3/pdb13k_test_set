import pandas as pd
from pprint import pprint

df = pd.read_pickle('./acetic_acid_all.pkl')
print(df)
pprint(df.columns.values)
