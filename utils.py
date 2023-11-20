from tqdm import tqdm
import numpy as np
import random
import torch
import os
import pandas as pd

def train_val_test_split(df, train_ratio, validate_ratio):
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_ratio * m)
    validate_end = int(validate_ratio * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test

def set_seed(seed, cudnn=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if (seed is not None) and cudnn:
        torch.backends.cudnn.deterministic = True

def train_one_epoch(train_df, learner, reg=True, batch_size=None):

    num_sample = len(train_df)
    if not batch_size:
        batch_size = num_sample

    perm = list(range(num_sample))
    random.shuffle(perm)

    start_idx = 0
    while start_idx + batch_size <= num_sample:
        batch_df = train_df.iloc[perm[start_idx:start_idx + batch_size]]

        learner.pareto_update(batch_df, reg)
        start_idx += batch_size

def train_learner(learner, data_df, train_ratio, val_ratio, pre_num_epochs, num_epochs):

    train_df, val_df, test_df = train_val_test_split(data_df, train_ratio, val_ratio)

    test_acc_obj_list, test_f_obj_list, test_acc_list, test_auc_list, test_g1dp_list, test_g1eo_list, test_g1eodds_list, test_g2dp_list, test_g2eo_list, test_g2eodds_list= [], [], [], [], [], [], [], [], [], []



    for epoch in tqdm(range(num_epochs)):
        train_one_epoch(train_df, learner)

        test_acc_obj, test_f_obj, test_acc, test_auc, test_g1dp, test_g1eo, test_g1eodds, test_g2dp, test_g2eo, test_g2eodds = learner.evaluate_model(test_df)
        test_acc_obj_list.append(test_acc_obj)
        test_f_obj_list.append(test_f_obj)
        test_acc_list.append(test_acc)
        test_auc_list.append(test_auc)
        test_g1dp_list.append(test_g1dp)
        test_g1eo_list.append(test_g1eo)
        test_g1eodds_list.append(test_g1eodds)
        test_g2dp_list.append(test_g2dp)
        test_g2eo_list.append(test_g2eo)
        test_g2eodds_list.append(test_g2eodds)

    return {
        "Test_acc_obj_list": test_acc_obj_list,
        "Test_f_obj_list": test_f_obj_list,
        "Test_acc_list": test_acc_list,
        "Test_auc_list": test_auc_list,
        "Test_g1dp_list": test_g1dp_list,
        "Test_g1eo_list": test_g1eo_list,
        "Test_g1eodds_list": test_g1eodds_list,
        "Test_g2dp_list": test_g2dp_list,
        "Test_g2eo_list": test_g2eo_list,
        "Test_g2eodds_list": test_g2eodds_list
    }

def load_dataset_adult():
    """ Loads adult dataset from preprocessed data file.

    Returns: 
      A pandas dataframe with all string features converted to binary one hot encodings.
    """
    cur_dir   = os.getcwd()
    data_path = os.path.join(cur_dir, "data/adult_processed.csv")
    df = pd.read_csv(data_path)
    return df

def load_dataset_compas():
    """ Loads adult dataset from preprocessed data file.

    Returns: 
      A pandas dataframe with all string features converted to binary one hot encodings.
    """
    cur_dir   = os.getcwd()
    data_path = os.path.join(cur_dir, "./data/Compas.csv")
    df = pd.read_csv(data_path)

    target_col = 'two_year_recid'
    feature_cols = [col for col in df.keys() if 'race' not in col and 'sex' not in col]
    feature_cols.remove(target_col)
    group_cols = ['race', 'sex_Female']
    return df, target_col, feature_cols, group_cols

def find_final_results(results_dict):
    idx = -1
    return (
        results_dict['Test_acc_list'][idx],
        results_dict['Test_auc_list'][idx],
        results_dict['Test_g1dp_list'][idx],
        results_dict['Test_g1eo_list'][idx],
        results_dict['Test_g1eodds_list'][idx],
        results_dict['Test_g2dp_list'][idx],
        results_dict['Test_g2eo_list'][idx],
        results_dict['Test_g2eodds_list'][idx],
    )