# %%
import pandas as pd
from sklearn.model_selection import train_test_split
# %%
RANDOM_SEED = 19237
# %%
# FOR ESOL
esol = pd.read_csv("../clustered_data/esol/esol.csv")
esol = esol.drop(columns=["quantile"])
train_esol, test_esol = train_test_split(esol, test_size=0.2, random_state=RANDOM_SEED, stratify=esol['cluster'])
train_esol.to_csv("../clustered_data/esol/train_esol.csv", index=False)
test_esol.to_csv("../clustered_data/esol/test_esol.csv", index=False)
# %%
# FOR QM9
qm9 = pd.read_csv("../clustered_data/qm9/qm9.csv")
qm9 = qm9.drop(columns=["mol", "quantile"])
train_qm9, test_qm9 = train_test_split(qm9, test_size=0.2, random_state=RANDOM_SEED, stratify=qm9['cluster'])
train_qm9, validation_qm9 = train_test_split(train_qm9, test_size=0.25, random_state=RANDOM_SEED, stratify=train_qm9['cluster']) # 60/20/20 split
train_qm9.to_csv("../clustered_data/qm9/train_qm9.csv", index=False)
validation_qm9.to_csv("../clustered_data/qm9/validation_qm9.csv", index=False)
test_qm9.to_csv("../clustered_data/qm9/test_qm9.csv", index=False)
# %%
hce = pd.read_csv("../clustered_data/hce/hce.csv")
hce
train_hce, test_hce = train_test_split(hce, test_size=0.2, random_state=RANDOM_SEED, stratify=hce['cluster'])
train_hce.to_csv("../clustered_data/hce/train_hce.csv", index=False)
test_hce.to_csv("../clustered_data/hce/test_hce.csv", index=False)
# %%