from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer


# ============================================================
# CONFIG
# ============================================================

TOKENIZER_PATH = Path(
    "models/tokenizer/tokenizer.json"
)

TRAIN_PATH = Path(
    "data/processed/openassistant_train.txt"
)

VALIDATION_PATH = Path(
    "data/processed/openassistant_validation.txt"
)

BLOCK_SIZE = 128
BATCH_SIZE = 8


# ============================================================
# DATASET
# ============================================================

class BPEDataset(Dataset):

    def __init__(
        self,
        token_ids,
        block_size
    ):

        self.tokens = token_ids
        self.block_size = block_size

        self.length = (
            len(self.tokens)
            - block_size
        )

    def __len__(self):

        return self.length

    def __getitem__(
        self,
        index
    ):

        x = self.tokens[
            index:
            index + self.block_size
        ]

        y = self.tokens[
            index + 1:
            index + self.block_size + 1
        ]

        x = torch.tensor(
            x,
            dtype=torch.long
        )

        y = torch.tensor(
            y,
            dtype=torch.long
        )

        return x, y


# ============================================================
# LOAD TOKENIZER
# ============================================================

def load_tokenizer():

    print(
        "Loading tokenizer..."
    )

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_PATH)
    )

    print(
        "Vocabulary:",
        tokenizer.get_vocab_size()
    )

    return tokenizer


# ============================================================
# ENCODE TEXT
# ============================================================

def encode_file(
    tokenizer,
    path
):

    print()

    print(
        "Encoding:",
        path
    )

    text = path.read_text(
        encoding="utf-8"
    )

    encoded = tokenizer.encode(
        text
    )

    token_ids = encoded.ids

    print(
        "Characters:",
        len(text)
    )

    print(
        "Tokens:",
        len(token_ids)
    )

    return token_ids


# ============================================================
# CREATE DATASET
# ============================================================

def create_dataset(
    token_ids,
    block_size
):

    dataset = BPEDataset(
        token_ids,
        block_size
    )

    print(
        "Dataset samples:",
        len(dataset)
    )

    return dataset


# ============================================================
# TEST
# ============================================================

def main():

    print("=" * 60)

    print(
        "MICHAEL AI - BPE DATASET TEST"
    )

    print("=" * 60)

    print()


    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = load_tokenizer()


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    train_tokens = encode_file(
        tokenizer,
        TRAIN_PATH
    )


    train_dataset = create_dataset(
        train_tokens,
        BLOCK_SIZE
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation_tokens = encode_file(
        tokenizer,
        VALIDATION_PATH
    )


    validation_dataset = create_dataset(
        validation_tokens,
        BLOCK_SIZE
    )


    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


    # --------------------------------------------------------
    # Get batch
    # --------------------------------------------------------

    x_batch, y_batch = next(
        iter(train_loader)
    )


    print()

    print("=" * 60)

    print(
        "BATCH TEST"
    )

    print("=" * 60)

    print()

    print(
        "Input shape:",
        x_batch.shape
    )

    print(
        "Target shape:",
        y_batch.shape
    )

    print()

    print(
        "Input:"
    )

    print(
        x_batch[0]
    )

    print()

    print(
        "Target:"
    )

    print(
        y_batch[0]
    )


    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    input_ids = (
        x_batch[0]
        .tolist()
    )

    target_ids = (
        y_batch[0]
        .tolist()
    )


    input_text = tokenizer.decode(
        input_ids
    )

    target_text = tokenizer.decode(
        target_ids
    )


    print()

    print(
        "INPUT TEXT:"
    )

    print(
        repr(input_text)
    )

    print()

    print(
        "TARGET TEXT:"
    )

    print(
        repr(target_text)
    )


    # --------------------------------------------------------
    # Alignment test
    # --------------------------------------------------------

    print()

    print("=" * 60)

    print(
        "SHIFT TEST"
    )

    print("=" * 60)

    print()

    print(
        "Input token 0:",
        x_batch[0][0].item()
    )

    print(
        "Input token 1:",
        x_batch[0][1].item()
    )

    print(
        "Target token 0:",
        y_batch[0][0].item()
    )

    print(
        "Target token 1:",
        y_batch[0][1].item()
    )


    if (
        x_batch[0][1].item()
        ==
        y_batch[0][0].item()
    ):

        print()

        print(
            "SHIFT TEST: PASS"
        )

    else:

        print()

        print(
            "SHIFT TEST: FAILED"
        )


    print()

    print("=" * 60)

    print(
        "BPE DATASET TEST FINISHED"
    )

    print("=" * 60)


if __name__ == "__main__":

    main()