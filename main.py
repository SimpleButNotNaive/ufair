import json
from utils import load_dataset_compas, set_seed, find_final_results, train_learner
from model import Fair_learner
import numpy as np

data_df, target_col, feature_cols, group_cols = load_dataset_compas()

pre_num_epoch = 0
num_epoch = 500
lr = 0.1
train_ratio = 0.6
val_ratio = 0.2
num_features = len(feature_cols)
seed = 100
alpha_lr = 0.1

accuracy_list = []
auc_list = []
g1dp_list = []
g1eo_list = []
g1eodds_list = []
g2dp_list = []
g2eo_list = []
g2eodds_list = []


for i in range(5):
	set_seed(seed + i)
	learner = Fair_learner(num_features, [50], feature_cols, target_col, group_cols, lr, alpha_lr)
	results_dict = train_learner(learner, data_df, train_ratio, val_ratio, pre_num_epoch, num_epoch)

	acc, auc, g1dp, g1eo, g1eodds, g2dp, g2eo, g2eodds = find_final_results(results_dict)
	accuracy_list.append(acc)
	auc_list.append(auc)
	g1dp_list.append(g1dp)
	g1eo_list.append(g1eo)
	g1eodds_list.append(g1eodds)

	g2dp_list.append(g2dp)
	g2eo_list.append(g2eo)
	g2eodds_list.append(g2eodds)


print(f"ACC: {np.array(accuracy_list).mean()*100:.1f}±{np.array(accuracy_list).std()*100:.1f}")
print(f"AUC: {np.array(auc_list).mean()     *100:.1f}±{np.array(auc_list).std()	   *100:.1f}")
print(f"G1_DP: {np.array(g1dp_list).mean()    *100:.1f}±{np.array(g1dp_list).std()    *100:.1f}")
print(f"G1_EO: {np.array(g1eo_list).mean()    *100:.1f}±{np.array(g1eo_list).std()    *100:.1f}")
print(f"G1_EOdds: {np.array(g1eodds_list).mean() *100:.1f}±{np.array(g1eodds_list).std() *100:.1f}")
print(f"G2_DP: {np.array(g2dp_list).mean()    *100:.1f}±{np.array(g2dp_list).std()    *100:.1f}")
print(f"G2_EO: {np.array(g2eo_list).mean()    *100:.1f}±{np.array(g2eo_list).std()    *100:.1f}")
print(f"G2_EOdds: {np.array(g2eodds_list).mean() *100:.1f}±{np.array(g2eodds_list).std() *100:.1f}")