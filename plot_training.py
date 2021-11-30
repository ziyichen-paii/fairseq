#!/usr/bin/env python3

import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

log_path = '/NAS5/speech/user/ziyichen/src/paii/fairseq/outputs/2021-11-19/14-27-11/hydra_train.log'

with open(log_path, 'r') as log_f:
    lines = log_f.readlines()

train_df = pd.DataFrame()
valid_df = pd.DataFrame()
for line in lines:
    match_train = re.search(r'\[train\]', line)
    match_valid = re.search(r'\[valid\]', line)
    if match_train:
        dict_string = re.search(r'\{.+\}$', line.strip('\n'))[0]
        train_dict = json.loads(dict_string)
        train_df = train_df.append(train_dict, ignore_index=True)
    elif match_valid:
        dict_string = re.search(r'\{.+\}$', line.strip('\n'))[0]
        valid_dict = json.loads(dict_string)
        valid_df = valid_df.append(valid_dict, ignore_index=True)


# train_df['group'] = 'train'
# train_df.rename(columns={'train_loss': 'loss', 'train_accuracy': 'accuracy'}, inplace=True)
# valid_df.rename(columns={'valid _loss': 'loss', 'valid_accuracy': 'accuracy'}, inplace=True)

# valid_df['group'] = 'valid'
valid_df = valid_df.drop_duplicates(subset='epoch', keep='last')
all_df = train_df.merge(valid_df, on='epoch', how='inner')

all_df = pd.concat([train_df[select_cols], valid_df[select_cols]], axis = 0)
all_df['epoch'] = all_df['epoch'].astype('int')
all_df['train_loss'] = all_df['train_loss'].astype('float')
all_df['valid_loss'] = all_df['valid_loss'].astype('float')
all_df['train_accuracy'] = all_df['train_accuracy'].astype('float')
all_df['valid_accuracy'] = all_df['valid_accuracy'].astype('float')

ax_loss = all_df.plot(x='epoch', y=['train_loss', 'valid_loss'])
fig = ax_loss.get_figure()
fig.savefig("loss.png")
ax_acc = all_df.plot(x='epoch', y=['train_accuracy', 'valid_accuracy'])
fig = ax_acc.get_figure()
fig.savefig("accuracy.png")


