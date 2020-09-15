import torch
import pandas as pd

from torch.utils.data import Dataset


def read_nsmc_examples(split):
    file_dir = {
        "train": "data/ratings_train.txt",
        "test": "data/ratings_test.txt"
    }

    if split.lower() not in file_dir:
        raise ValueError(f"Invalid data type \"{split}\"")

    df = pd.read_csv(file_dir[split.lower()], sep='\t', encoding='utf-8')
    df.dropna(inplace=True)

    examples = []
    for example in list(zip(df["id"], df["document"], df["label"].values)):
        uid, text, label = example
        examples.append(dict(uid=uid, text=text, label=label))

    return examples


class NSMCDataSet(Dataset):
    def __init__(self, data_split, tokenizer, max_seq_length=512, pad_to_max=False):

        self.tokenizer = tokenizer
        self.bos_token_id = 3
        self.eos_token_id = 4
        self.pad_token_id = 0

        self.max_seq_length = max_seq_length
        self.pad_to_max = pad_to_max

        self.examples = read_nsmc_examples(data_split)
        self.features = self._featurize()

    def _featurize(self):
        features = []
        for example in self.examples:
            tokens, _ = self.tokenizer.tokenize(example["text"])
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            features.append(dict(input_ids=input_ids, label=example["label"]))

        return features

    def __getitem__(self, index):
        return self.features[index]

    def __len__(self):
        return len(self.features)

    def collate_fn(self, batch):
        all_input_ids = []
        all_attention_mask = []
        all_labels = []

        lengths = []
        for feature in batch:
            input_ids = feature["input_ids"]

            if len(input_ids) > (self.max_seq_length-2):
                input_ids = input_ids[:(self.max_seq_length-2)]
            input_ids = [self.bos_token_id] + input_ids + [self.eos_token_id]
            attention_mask = [1] * len(input_ids)

            if self.pad_to_max:
                pad_length = self.max_seq_length - len(input_ids)
                input_ids.extend([self.pad_token_id] * pad_length)
                attention_mask.extend([0] * pad_length)
            else:
                lengths.append(len(input_ids))

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(feature["label"])

        if not self.pad_to_max:
            max_seq_length_in_batch = max(lengths)

            for i in range(len(batch)):
                pad_length = max_seq_length_in_batch - len(all_input_ids[i])
                all_input_ids[i].extend([self.pad_token_id] * pad_length)
                all_attention_mask[i].extend([0] * pad_length)

        return all_input_ids, all_attention_mask, all_labels
