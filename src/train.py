from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.tokenizer import CharacterTokenizer
from src.model import MichaelAI


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = Path("data/dataset.txt")

CHECKPOINT_DIR = Path("checkpoints")

BEST_MODEL_PATH = (
    CHECKPOINT_DIR / "best_model.pt"
)

LATEST_MODEL_PATH = (
    CHECKPOINT_DIR / "latest_model.pt"
)

BLOCK_SIZE = 32

BATCH_SIZE = 4

EMBED_DIM = 128

NUM_HEADS = 4

NUM_LAYERS = 2

HIDDEN_DIM = 512

LEARNING_RATE = 3e-4

EPOCHS = 100

TRAIN_RATIO = 0.90


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cpu")


# ============================================================
# LANGUAGE MODEL DATASET
# ============================================================

class LanguageModelDataset(Dataset):

    def __init__(
        self,
        tokens,
        block_size
    ):

        self.tokens = tokens

        self.block_size = block_size

        self.length = (
            len(tokens)
            - block_size
        )


    def __len__(self):

        return self.length


    def __getitem__(self, index):

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
# CALCULATE LOSS
# ============================================================

def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0

    batch_count = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(DEVICE)

            y = y.to(DEVICE)

            logits = model(x)

            batch_size, seq_len, vocab_size = (
                logits.shape
            )

            logits = logits.view(
                batch_size * seq_len,
                vocab_size
            )

            y = y.view(
                batch_size * seq_len
            )

            loss = criterion(
                logits,
                y
            )

            total_loss += loss.item()

            batch_count += 1

    return total_loss / batch_count


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "MICHAEL AI - TRAIN / VALIDATION"
    )

    print("=" * 60)


    # ========================================================
    # DEVICE
    # ========================================================

    print()

    print(
        "Device:",
        DEVICE
    )


    # ========================================================
    # LOAD DATA
    # ========================================================

    print()

    print(
        "Loading dataset..."
    )

    text = DATASET_PATH.read_text(
        encoding="utf-8"
    )

    print(
        "Total characters:",
        len(text)
    )


    # ========================================================
    # TOKENIZER
    # ========================================================

    tokenizer = CharacterTokenizer(
        text
    )

    vocab_size = tokenizer.vocab_size

    print(
        "Vocabulary size:",
        vocab_size
    )


    # ========================================================
    # ENCODE DATASET
    # ========================================================

    tokens = tokenizer.encode(
        text
    )


    # ========================================================
    # TRAIN / VALIDATION SPLIT
    # ========================================================

    split_index = int(
        len(tokens) * TRAIN_RATIO
    )

    train_tokens = tokens[
        :split_index
    ]

    validation_tokens = tokens[
        split_index:
    ]


    print()

    print(
        "Train tokens:",
        len(train_tokens)
    )

    print(
        "Validation tokens:",
        len(validation_tokens)
    )


    # ========================================================
    # DATASET OBJECTS
    # ========================================================

    train_dataset = LanguageModelDataset(
        train_tokens,
        BLOCK_SIZE
    )

    validation_dataset = LanguageModelDataset(
        validation_tokens,
        BLOCK_SIZE
    )


    print()

    print(
        "Train samples:",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(validation_dataset)
    )


    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


    # ========================================================
    # MODEL
    # ========================================================

    model = MichaelAI(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        hidden_dim=HIDDEN_DIM,
        block_size=BLOCK_SIZE
    )

    model = model.to(DEVICE)


    # ========================================================
    # PARAMETER COUNT
    # ========================================================

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print()

    print(
        "Model parameters:",
        f"{total_parameters:,}"
    )


    # ========================================================
    # LOSS
    # ========================================================

    criterion = nn.CrossEntropyLoss()


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )


    # ========================================================
    # CHECKPOINT DIRECTORY
    # ========================================================

    CHECKPOINT_DIR.mkdir(
        exist_ok=True
    )


    # ========================================================
    # BEST VALIDATION LOSS
    # ========================================================

    best_validation_loss = float(
        "inf"
    )


    # ========================================================
    # TRAINING
    # ========================================================

    print()

    print("=" * 60)

    print(
        "START TRAINING"
    )

    print("=" * 60)


    for epoch in range(EPOCHS):

        model.train()

        total_train_loss = 0.0

        train_batch_count = 0


        # ====================================================
        # TRAIN
        # ====================================================

        for x, y in train_loader:

            x = x.to(DEVICE)

            y = y.to(DEVICE)


            # Reset gradients
            optimizer.zero_grad()


            # Forward
            logits = model(x)


            batch_size, seq_len, vocab_size = (
                logits.shape
            )


            # Flatten
            logits = logits.view(
                batch_size * seq_len,
                vocab_size
            )

            y = y.view(
                batch_size * seq_len
            )


            # Loss
            loss = criterion(
                logits,
                y
            )


            # Backpropagation
            loss.backward()


            # Update weights
            optimizer.step()


            total_train_loss += (
                loss.item()
            )

            train_batch_count += 1


        # ====================================================
        # TRAIN LOSS
        # ====================================================

        train_loss = (
            total_train_loss
            / train_batch_count
        )


        # ====================================================
        # VALIDATION LOSS
        # ====================================================

        validation_loss = evaluate(
            model,
            validation_loader,
            criterion
        )


        # ====================================================
        # PRINT
        # ====================================================

        print(
            f"Epoch {epoch + 1:03d}/{EPOCHS} "
            f"| Train Loss: {train_loss:.4f} "
            f"| Val Loss: {validation_loss:.4f}"
        )


        # ====================================================
        # SAVE LATEST
        # ====================================================

        torch.save(
            {
                "epoch": epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "train_loss":
                    train_loss,

                "validation_loss":
                    validation_loss,

                "vocab_size":
                    vocab_size,

                "block_size":
                    BLOCK_SIZE,

                "embed_dim":
                    EMBED_DIM,

                "num_heads":
                    NUM_HEADS,

                "num_layers":
                    NUM_LAYERS,

                "hidden_dim":
                    HIDDEN_DIM
            },

            LATEST_MODEL_PATH
        )


        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if validation_loss < best_validation_loss:

            best_validation_loss = (
                validation_loss
            )

            torch.save(
                {
                    "epoch":
                        epoch + 1,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "train_loss":
                        train_loss,

                    "validation_loss":
                        validation_loss,

                    "vocab_size":
                        vocab_size,

                    "block_size":
                        BLOCK_SIZE,

                    "embed_dim":
                        EMBED_DIM,

                    "num_heads":
                        NUM_HEADS,

                    "num_layers":
                        NUM_LAYERS,

                    "hidden_dim":
                        HIDDEN_DIM
                },

                BEST_MODEL_PATH
            )


            print(
                "  -> Best model saved."
            )


    # ========================================================
    # FINISHED
    # ========================================================

    print()

    print("=" * 60)

    print(
        "TRAINING FINISHED"
    )

    print("=" * 60)

    print()

    print(
        "Best validation loss:",
        best_validation_loss
    )

    print()

    print(
        "Best model:",
        BEST_MODEL_PATH
    )

    print(
        "Latest model:",
        LATEST_MODEL_PATH
    )


if __name__ == "__main__":

    main()