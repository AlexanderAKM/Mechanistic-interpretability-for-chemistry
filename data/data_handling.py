# %%
import pandas as pd
from sklearn.model_selection import train_test_split
# %%
RANDOM_SEED = 19237
# %%
esol = pd.read_csv("ESOL.csv")
esol = esol.drop(columns=[column for column in esol.columns.to_list() if column not in ["measured log solubility in mols per litre", "smiles"]])
length_esol = len(esol)
train_esol, test_esol = train_test_split(esol, test_size=0.2, random_state=RANDOM_SEED)
train_esol.to_csv("train_ESOL.csv", index=False)
test_esol.to_csv("test_ESOL.csv", index=False)
# %%
qm9 = pd.read_csv("qm9.csv")
qm9 = qm9.sample(length_esol, random_state=RANDOM_SEED)
qm9 = qm9.drop(columns=[column for column in qm9.columns.to_list() if column not in ["smiles", "g298_atom"]])
train_qm9, test_qm9 = train_test_split(qm9, test_size=0.2, random_state=RANDOM_SEED)
train_qm9.to_csv("train_qm9.csv", index=False)
test_qm9.to_csv("test_qm9.csv", index=False)
# %%
hce = pd.read_csv("hce.csv")
hce = hce.sample(length_esol, random_state=RANDOM_SEED)
hce = hce.drop(columns=[column for column in hce.columns.to_list() if column not in ["smiles", "pce_1"]])
train_hce, test_hce = train_test_split(hce, test_size=0.2, random_state=RANDOM_SEED)
train_hce.to_csv("train_hce.csv", index=False)
test_hce.to_csv("test_hce.csv", index=False)
# %%
