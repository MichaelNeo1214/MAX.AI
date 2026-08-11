from pathlib import Path

import torch

from src.tokenizer import CharacterTokenizer
from src.model import MichaelAI


def main():

    print("=" * 60)
    print("MICHAEL AI - MODEL TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    dataset_path = Path(
        "data/dataset.txt"
    )

    text = dataset_path.read_text(
        encoding="utf-8"
    )

    # ---------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------

    tokenizer = CharacterTokenizer(
        text
    )

    vocab_size = tokenizer.vocab_size

    print()
    print("Vocabulary size:", vocab_size)

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = MichaelAI(
        vocab_size=vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        hidden_dim=512,
        block_size=32
    )

    # ---------------------------------------------------------
    # Parameter count
    # ---------------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        "Total parameters:",
        f"{total_parameters:,}"
    )

    # ---------------------------------------------------------
    # Test input
    # ---------------------------------------------------------

    sample_text = "Nama saya Michael."

    encoded = tokenizer.encode(
        sample_text
    )

    input_ids = torch.tensor(
        [encoded],
        dtype=torch.long
    )

    print()
    print("Input shape:")
    print(input_ids.shape)

    # ---------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------

    with torch.no_grad():

        logits = model(
            input_ids
        )

    print()
    print("Output shape:")
    print(logits.shape)

    print()
    print("Model berhasil melakukan forward pass.")


if __name__ == "__main__":
    main()