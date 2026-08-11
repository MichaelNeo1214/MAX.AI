from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


# ============================================================
# CONFIG
# ============================================================

TRAIN_DATA = Path(
    "data/processed/openassistant_train.txt"
)

VALIDATION_DATA = Path(
    "data/processed/openassistant_validation.txt"
)

OUTPUT_DIR = Path(
    "models/tokenizer"
)

OUTPUT_FILE = (
    OUTPUT_DIR / "tokenizer.json"
)

VOCAB_SIZE = 16000


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MICHAEL AI - BPE TOKENIZER TRAINING")
    print("=" * 60)

    print()

    print(
        "Training data:",
        TRAIN_DATA
    )

    print(
        "Validation data:",
        VALIDATION_DATA
    )

    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Create BPE tokenizer
    # --------------------------------------------------------

    tokenizer = Tokenizer(
        BPE(
            unk_token="<UNK>"
        )
    )


    tokenizer.pre_tokenizer = Whitespace()


    # --------------------------------------------------------
    # Special tokens
    # --------------------------------------------------------

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,

        min_frequency=2,

        special_tokens=[
            "<PAD>",
            "<UNK>",
            "<BOS>",
            "<EOS>",
            "<USER>",
            "<ASSISTANT>"
        ]
    )


    # --------------------------------------------------------
    # Train tokenizer
    # --------------------------------------------------------

    print(
        "Training tokenizer..."
    )

    tokenizer.train(
        [
            str(TRAIN_DATA)
        ],
        trainer=trainer
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    tokenizer.save(
        str(OUTPUT_FILE)
    )


    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    print()

    print(
        "Tokenizer vocabulary:",
        tokenizer.get_vocab_size()
    )

    print()

    test_text = (
        "Artificial intelligence "
        "adalah teknologi komputer."
    )

    encoded = tokenizer.encode(
        test_text
    )


    print(
        "TEST TEXT:"
    )

    print(
        test_text
    )

    print()

    print(
        "TOKENS:"
    )

    print(
        encoded.tokens
    )

    print()

    print(
        "TOKEN IDS:"
    )

    print(
        encoded.ids
    )

    print()

    print("=" * 60)

    print(
        "TOKENIZER TRAINING FINISHED"
    )

    print("=" * 60)

    print()

    print(
        "Saved:",
        OUTPUT_FILE
    )


if __name__ == "__main__":

    main()