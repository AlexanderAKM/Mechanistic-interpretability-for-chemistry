This scripts/ folder is organized roughly as following, and should allow anyone to completely copy replicate the results of the paper.

Firstly, in @load_data.py the data for qm9, esol and hce are retrieved from online and clustered accordingly to utils/clustering.py and for hce based on the property quantiles. Secondly, the data is split in @data_splitting.py. After that, the models are trained in training.py. In TL_chem.py all of the mechanistic interpretability techniques are done for all of the models.

