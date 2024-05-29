import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import DataCollatorForLanguageModeling, EsmForMaskedLM, EsmTokenizer, EsmConfig
from typing import Any, Optional, Tuple
from Bio import SeqIO

SEED = 43
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)


class DataCollatorForFIM(DataCollatorForLanguageModeling):
    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 100% of middle part is masked with [MASK].
        For this custom collator, only the middle part of the sequences is masked.
        """
        labels = inputs.clone()
        batch_size, seq_len = labels.shape

        split_points = torch.sort(torch.randint(1, seq_len - 1, (batch_size, 2)), dim=-1)[0]
        prefix_lengths = split_points[:, 0]
        suffix_lengths = split_points[:, 1]

        arange = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        prefix_mask = arange < prefix_lengths.unsqueeze(1)
        suffix_mask = arange >= suffix_lengths.unsqueeze(1)

        non_middle_mask = prefix_mask | suffix_mask

        labels[non_middle_mask] = -100

        mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        inputs[~non_middle_mask] = mask_token_id

        return inputs, labels


class EsmDataset(Dataset):
    def __init__(self, file_path, tokenizer, chunk_size=254):
        sequences = [str(record.seq) for record in SeqIO.parse(file_path, "fasta")]
        eos_token = tokenizer.eos_token
        all_text = eos_token.join(sequences)

        self.encoding = tokenizer(
            all_text,
            truncation=False,
            max_length=None,
            padding=False,
            add_special_tokens=False
        )

        self.chunk_size = chunk_size
        self.num_chunks = len(self.encoding['input_ids']) // self.chunk_size

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        start_index = idx * self.chunk_size
        end_index = start_index + self.chunk_size

        chunked_input_ids = self.encoding['input_ids'][start_index:end_index]
        attention_mask = [1] * len(chunked_input_ids)

        return {"input_ids": torch.tensor(chunked_input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long)}


def make_esm_dataset(file_path, tokenizer, chunk_size=254):
    full_dataset = EsmDataset(file_path, tokenizer, chunk_size=chunk_size)
    total_length = len(full_dataset)
    train_size = int(0.9 * total_length)
    remaining_size = total_length - train_size
    val_size = remaining_size // 2
    test_size = remaining_size - val_size
    train_dataset, remainder = random_split(full_dataset, [train_size, remaining_size])
    val_dataset, test_dataset = random_split(remainder, [val_size, test_size])

    return train_dataset, val_dataset, test_dataset

def process_raw(file_path, test_size=0.1):
    with open(file_path, "r") as file:
        lines = file.readlines()
        sequences = [seq.strip() for seq in lines if not seq.startswith(">")]
        df = pd.DataFrame({'sequence': sequences})
    train_data, test_data = train_test_split(df, test_size=test_size, shuffle=True, random_state=42)
    test_data, val_data = train_test_split(test_data, test_size=(1 - test_size) / 2, shuffle=True, random_state=42)
    return train_data, val_data, test_data


def process_for_val_batch(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
        sequences = [seq.strip() for seq in lines if not seq.startswith(">")]
    return sequences


def process_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    output = []

    for line in lines:
        line = line.strip()
        n = len(line)

        idx1, idx2 = torch.randperm(n - 2)[:2] + 1
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1

        prefix, middle, suffix = line[:idx1], line[idx1:idx2], line[idx2:]

        p = rng.random()
        if p > 0.75:  # PSM
            fim_sample = '@' + prefix + '$' + suffix + '#' + middle
        elif p > 0.5:  # SPM
            # fim_sample = '$' + suffix + '@' + prefix + '#' + middle
            fim_sample = '@' + '$' + suffix + '#' + prefix + middle  # SPM v2
        else:  # default
            fim_sample = prefix + middle + suffix

        output.append(fim_sample)

    return '.'.join(output)
