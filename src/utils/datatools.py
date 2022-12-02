import numpy as np, pandas as pd, argparse, sys, re, os, json
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def split_graph_dataset(raw_data_path, proc_data_path):
	if not os.path.isdir(proc_data_path):
		os.mkdir(proc_data_path)

	split_df = pd.read_csv(os.path.join(raw_data_path, 'split.csv'))
	split_df.set_index('id', inplace=True)
	split_names = split_df['split'].unique()

	labels_df = pd.read_csv(os.path.join(raw_data_path, 'label.csv'))
	labels_df.set_index('id', inplace=True)

	edges_df = pd.read_csv(os.path.join(raw_data_path, 'edge.csv'))
	posts_series = edges_df[edges_df.relation=='post'].groupby('source_id')['target_id'].apply(list)
	friends_series = edges_df[edges_df.relation=='friend'].groupby('source_id')['target_id'].apply(list)
	follows_series = edges_df[edges_df.relation=='follow'].groupby('source_id')['target_id'].apply(list)
	del edges_df

	with open(os.path.join(raw_data_path, 'node.json')) as f:
		txt = f.read()
		all_nodes = json.loads(txt)

	user_dicts, tweet_dict = {}, {}

	while len(all_nodes)>0:

		node = all_nodes.pop()

		if node['id'][0]=='u':
			user_dicts[node['id']] = node
		elif node['id'][0]=='t':
			tweet_dict[node['id']] = node['text']

	del all_nodes

	user_ids = list(user_dicts.keys())
	user_splits = {split_name : [] for split_name in split_names}

	for user_id in tqdm(user_ids):

		node = user_dicts[user_id]
		split_name = split_df.loc[user_id].split

		try:
			label_name = labels_df.loc[user_id].label
		except KeyError:
			label_name = None

		node['label'] = label_name
		node['tweets'] = []

		try:
			posts = posts_series[user_id]
		except KeyError:
			posts = []

		for tweet_id in posts:
			node['tweets'].append(tweet_dict.pop(tweet_id))

		try:
			node['friends'] = friends_series[user_id]
		except KeyError:
			node['friends'] = []

		try:
			node['follows'] = follows_series[user_id]
		except KeyError:
			node['follows'] = []

		user_splits[split_name].append(node)
		user_dicts.pop(user_id)


	for split_name in split_names:
		with open(os.path.join(proc_data_path, split_name + '.json'), 'w') as f:
			json.dump(user_splits[split_name],f)

	return


class DescriptionOnlyDatast(Dataset):
	def __init__(self, dataset, args, tokenizer):
        self.dataset = dataset
        self.p = args
        self.tokenizer = tokenizer

    def __len__(self):
    	return len(self.dataset)

   	def __getitem__(self, idx):
        ele = self.dataset[idx]
        return ele

    def pad_data(self, data):
        sents = [x['description'] for x in data]
        labels = [x['label'] for x in data]
        uids = [x['id'] for x in data]
        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        token_type_ids = torch.LongTensor(encoding['token_type_ids'])
        labels = torch.LongTensor(labels)

        return token_ids, token_type_ids, attention_mask, labels, sents, uids

    def collate_fn(self, all_data):

        token_ids, token_type_ids, attention_mask, labels, sents, uids = self.pad_data(all_data)
        batched_data = {
                'token_ids': token_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'sents': sents,
                'account_id': uids
            }

        return batched_data


if __name__ == "__main__":
	split_graph_dataset(raw_data_path='data/twibot20/raw', proc_data_path='data/twibot20/processed')
