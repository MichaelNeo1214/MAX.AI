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

VALIDATION_PATH = Path(
    "data/processed/openassistant_validation.txt"
)

CHECKPOINT_DIR = Path(
    "checkpoints/bpe"
)

BEST_MODEL_PATH = (
    CHECKPOINT_DIR / "best_model.pt"
)

LATEST_MODEL_PATH = (
    CHECKPOINT_DIR / "latest_model.pt"
)


# ============================================================
# MODEL
# ============================================================

BLOCK_SIZE = 128

BATCH_SIZE = 8

EMBED_DIM = 128

NUM_HEADS = 4

NUM_LAYERS = 2

HIDDEN_DIM = 512


# ============================================================
# TRAINING
# ============================================================

LEARNING_RATE = 3e-4

EPOCHS = 1

GRAD_CLIP = 1.0


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cpu"
)


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0

    batch_count = 0

    for x, y in loader:

        x = x.to(DEVICE)

        y = y.to(DEVICE)

        logits = model(x)

        batch_size, seq_len, vocab_size = (
            logits.shape
        )

        logits = logits.reshape(
            batch_size * seq_len,
            vocab_size
        )

        y = y.reshape(
            batch_size * seq_len
        )

        loss = criterion(
            logits,
            y
        )

        total_loss += loss.item()

        batch_count += 1

    return (
        total_loss /
        batch_count
    )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    train_loss,
    validation_loss,
    vocab_size
):

    torch.save(
        {
            "epoch": epoch,

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
                HIDDEN_DIM,
        },
        path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "MICHAEL AI - BPE TRAINING"
    )

    print("=" * 60)

    print()

    print(
        "Device:",
        DEVICE
    )


    # ========================================================
    # TOKENIZER
    # ========================================================

    print()

    tokenizer = load_tokenizer()

    vocab_size = (
        tokenizer.get_vocab_size()
    )

    print()

    print(
        "Vocabulary:",
        vocab_size
    )


    # ========================================================
    # ENCODE TRAIN
    # ========================================================

    train_tokens = encode_file(
        tokenizer,
        TRAIN_PATH
    )


    # ========================================================
    # ENCODE VALIDATION
    # ========================================================

    validation_tokens = encode_file(
        tokenizer,
        VALIDATION_PATH
    )


    # ========================================================
    # DATASET
    # ========================================================

    train_dataset = BPEDataset(
        train_tokens,
        BLOCK_SIZE
    )

    validation_dataset = BPEDataset(
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
    # DATALOADER
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


    model = model.to(
        DEVICE
    )


    # ========================================================
    # PARAMETERS
    # ========================================================

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
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
    # CHECKPOINT
    # ========================================================

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    best_validation_loss = float(
        "inf"
    )


    # ========================================================
    # TRAIN
    # ========================================================

    print()

    print("=" * 60)

    print(
        "START TRAINING"
    )

    print("=" * 60)

    print()


    for epoch in range(
        EPOCHS
    ):

        model.train()

        total_train_loss = 0.0

        batch_count = 0


        for x, y in train_loader:

            x = x.to(
                DEVICE
            )

            y = y.to(
                DEVICE
            )


            # ------------------------------------------------
            # Reset gradients
            # ------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )


            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            logits = model(
                x
            )


            batch_size, seq_len, vocab_size = (
                logits.shape
            )


            # ------------------------------------------------
            # Flatten
            # ------------------------------------------------

            logits = logits.reshape(
                batch_size * seq_len,
                vocab_size
            )

            y = y.reshape(
                batch_size * seq_len
            )


            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = criterion(
                logits,
                y
            )


            # ------------------------------------------------
            # Backprop
            # ------------------------------------------------

            loss.backward()


            # ------------------------------------------------
            # Gradient clipping
            # ------------------------------------------------

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP
            )


            # ------------------------------------------------
            # Optimizer
            # ------------------------------------------------

            optimizer.step()


            total_train_loss += (
                loss.item()
            )

            batch_count += 1


            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if batch_count % 500 == 0:

                current_loss = (
                    total_train_loss /
                    batch_count
                )

                print(
                    f"Epoch {epoch + 1:03d} "
                    f"| Step {batch_count:,} "
                    f"| Train Loss: "
                    f"{current_loss:.4f}"
                )


        # ====================================================
        # TRAIN LOSS
        # ====================================================

        train_loss = (
            total_train_loss /
            batch_count
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        print()

        print(
            "Running validation..."
        )

        validation_loss = evaluate(
            model,
            validation_loader,
            criterion
        )


        # ====================================================
        # RESULTS
        # ====================================================

        print()

        print(
            f"Epoch {epoch + 1:03d}/{EPOCHS} "
            f"| Train Loss: {train_loss:.4f} "
            f"| Val Loss: {validation_loss:.4f}"
        )


        # ====================================================
        # SAVE LATEST
        # ====================================================

        save_checkpoint(
            LATEST_MODEL_PATH,

            model,

            optimizer,

            epoch + 1,

            train_loss,

            validation_loss,

            vocab_size
        )


        # ====================================================
        # SAVE BEST
        # ====================================================

        if validation_loss < best_validation_loss:

            best_validation_loss = (
                validation_loss
            )

            save_checkpoint(
                BEST_MODEL_PATH,

                model,

                optimizer,

                epoch + 1,

                train_loss,

                validation_loss,

                vocab_size
            )

            print()

            print(
                "Best model saved."
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
        "Train loss:",
        train_loss
    )

    print(
        "Validation loss:",
        validation_loss
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