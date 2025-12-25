import pandas as pd 
from datasail.sail import datasail
import json

df = pd.read_csv('src/data/chembl_data_subsample.csv')
df = df.sample(n=100000, random_state=42)
e_splits, _, _ = datasail(
    techniques=["C1e"],
    splits=[7, 1, 1],
    names=["train","val", "test"],
    runs=1,
    solver="SCIP",
    e_type="M",
    threads=71,
    e_data=dict(df[["mol_id", "smiles"]].values.tolist())
)

with open('src/data/data_split_100k.json', 'w') as f:
    json.dump(e_splits['C1e'], f)
