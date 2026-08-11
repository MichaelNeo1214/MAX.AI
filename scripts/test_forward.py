from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model import MichaelAI
from src.bpe_dataset import (
    load_tokenizer,
    encode_file,
    BPEDataset,
)


# ============================================================
# CONFIG
# ============================================================

TRAIN_PATH = Path(
    "data/processed/openassistant_train.txt"
)

TOKENIZER_PATH = Path(
    "models/tokenizer/tokenizer.json"
)

BLOCK_SIZE = 128
BATCH_SIZE = 2

EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
HIDDEN_DIM = 512

DEVICE = torch.device("cpu")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MICHAEL AI - TRANSFORMER FORWARD TEST")
    print("=" * 60)

    print()
    print("Device:", DEVICE)

    # --------------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------------

    tokenizer = load_tokenizer()

    vocab_size = tokenizer.get_vocab_size()

    print()
    print("Vocabulary:", vocab_size)

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    tokens = encode_file(
        tokenizer,
        TRAIN_PATH
    )

    # Untuk test kita tidak perlu seluruh dataset.
    # Ambil sebagian kecil saja.
    test_tokens = tokens[:5000]

    dataset = BPEDataset(
        test_tokens,
        BLOCK_SIZE
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    x, y = next(
        iter(loader)
    )

    x = x.to(DEVICE)
    y = y.to(DEVICE)

    print()
    print("Input shape :", x.shape)
    print("Target shape:", y.shape)

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = MichaelAI(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        hidden_dim=HIDDEN_DIM,
        block_size=BLOCK_SIZE
    )

    model = model.to(DEVICE)

    # --------------------------------------------------------
    # PARAMETER
    # --------------------------------------------------------

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print()
    print(
        "Model parameters:",
        f"{parameters:,}"
    )

    # --------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------

    print()
    print("Running forward pass...")

    with torch.no_grad():

        logits = model(x)

    print()
    print("Logits shape:", logits.shape)

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    batch_size, seq_len, vocab = logits.shape

    loss = criterion(
        logits.reshape(
            batch_size * seq_len,
            vocab
        ),
        y.reshape(
            batch_size * seq_len
        )
    )

    print()
    print(
        "Initial loss:",
        loss.item()
    )

    # --------------------------------------------------------
    # CHECK
    # --------------------------------------------------------

    expected_shape = (
        BATCH_SIZE,
        BLOCK_SIZE,
        vocab_size
    )

    print()

    if tuple(logits.shape) == expected_shape:

        print(
            "LOGITS SHAPE: PASS"
        )

    else:

        print(
            "LOGITS SHAPE: FAILED"
        )

    if torch.isfinite(loss):

        print(
            "LOSS TEST: PASS"
        )

    else:

        print(
            "LOSS TEST: FAILED"
        )

    print()
    print("=" * 60)
    print("FORWARD TEST FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()