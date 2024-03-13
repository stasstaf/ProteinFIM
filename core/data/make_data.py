import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)


def process_raw(file_path, test_size=0.75):
    with open(file_path, "r") as file:
        lines = file.readlines()
        sequences = [seq.strip() for seq in lines if not seq.startswith(">")]
        df = pd.DataFrame({'sequence': sequences})
    train_data, test_data = train_test_split(df, test_size=test_size, shuffle=True, random_state=42)
    test_data, val_data = train_test_split(test_data, test_size=(1-test_size)/2, shuffle=True, random_state=42)
    return train_data, val_data, test_data


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
        if p > 0.66:  # PSM
            fim_sample = '@' + prefix + '$' + suffix + '#' + middle
        elif p > 0.33:  # SPM
            fim_sample = '$' + suffix + '@' + prefix + '#' + middle
        else:  # default
            fim_sample = prefix + middle + suffix

        output.append(fim_sample)

    return '.'.join(output)