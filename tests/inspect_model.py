from pathlib import Path

from src.tokenizer import CharacterTokenizer
from src.model import MichaelAI

def main():

    text = Path(
        "data/dataset.txt"
    ).read_text(
        encoding="utf-8"
    )

    tokenizer = CharacterTokenizer(
        text
    )

    model = MichaelAI(
        vocab_size=tokenizer.vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        hidden_dim=512,
        block_size=32
    )

    print("=" * 60)
    print("MICHAEL AI - MODEL PARAMETERS")
    print("=" * 60)

    total = 0

    for name, parameter in model.named_parameters():

        count = parameter.numel()

        total += count

        print(
            f"{name:45}"
            f"{str(tuple(parameter.shape)):20}"
            f"{count:,}"
        )

    print()
    print(
        "TOTAL PARAMETERS:",
        f"{total:,}"
    )


if __name__ == "__main__":
    main()