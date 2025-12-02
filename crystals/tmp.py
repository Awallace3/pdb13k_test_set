import pandas as pd
from pprint import pprint

df = pd.read_pickle("./sapt0_induction/formamide_sapt0adz.pkl")
print(df)
pprint(df.columns.tolist())
