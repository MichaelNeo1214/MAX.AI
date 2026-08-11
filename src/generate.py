from pathlib import Path

import torch

from src.tokenizer import CharacterTokenizer
from src.model import MichaelAI


# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = Path("data/dataset.txt")
CHECKPOINT_PATH = Path(
    "checkpoints/model_epoch_100.pt"
)

BLOCK_SIZE = 32

DEVICE = torch.device("cpu")


# ============================================================
# GENERATE
# ============================================================

def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=0.8
):

    model.eval()

    # Prompt -> token IDs
    tokens = tokenizer.encode(prompt)

    input_ids = torch.tensor(
        [tokens],
        dtype=torch.long,
        device=DEVICE
    )

    with torch.no_grad():

        for _ in range(max_new_tokens):

            # Jangan melebihi context window
            input_context = input_ids[
                :, -BLOCK_SIZE:
            ]

            # Model prediction
            logits = model(
                input_context
            )

            # Ambil token terakhir
            logits = logits[:, -1, :]

            # Temperature
            logits = logits / temperature

            # Convert logits -> probability
            probabilities = torch.softmax(
                logits,
                dim=-1
            )

            # Sample token berikutnya
            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            # Tambahkan token ke sequence
            input_ids = torch.cat(
                [
                    input_ids,
                    next_token
                ],
                dim=1
            )

    # Token IDs -> text
    generated_text = tokenizer.decode(
        input_ids[0].tolist()
    )

    return generated_text


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MICHAEL AI - TEXT GENERATION")
    print("=" * 60)


    # ========================================================
    # LOAD DATASET
    # ========================================================

    text = DATASET_PATH.read_text(
        encoding="utf-8"
    )


    # ========================================================
    # TOKENIZER
    # ========================================================

    tokenizer = CharacterTokenizer(
        text
    )


    # ========================================================
    # CREATE MODEL
    # ========================================================

    model = MichaelAI(
        vocab_size=tokenizer.vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        hidden_dim=512,
        block_size=BLOCK_SIZE
    )


    # ========================================================
    # LOAD TRAINED WEIGHTS
    # ========================================================

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)


    print()
    print(
        "Loaded checkpoint:",
        CHECKPOINT_PATH
    )

    print(
        "Training loss:",
        checkpoint["loss"]
    )


    # ========================================================
    # GENERATION
    # ========================================================

    prompt = "Saya sedang"

    result = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=100,
        temperature=0.5
    )


    print()
    print("=" * 60)
    print("GENERATED TEXT")
    print("=" * 60)

    print()
    print(result)


if __name__ == "__main__":
    main()