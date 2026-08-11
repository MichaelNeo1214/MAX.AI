from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.tokenizer import CharacterTokenizer


class TextDataset:

    def __init__(
        self,
        text,
        tokenizer,
        block_size
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size

        # Text -> token IDs
        self.tokens = tokenizer.encode(text)

        # Jumlah sample yang tersedia
        self.length = len(self.tokens) - block_size

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        # Input
        x = self.tokens[
            index:
            index + self.block_size
        ]

        # Target digeser satu token
        y = self.tokens[
            index + 1:
            index + self.block_size + 1
        ]

        # Ubah menjadi Tensor PyTorch
        x = torch.tensor(
            x,
            dtype=torch.long
        )

        y = torch.tensor(
            y,
            dtype=torch.long
        )

        return x, y


def main():

    # =========================================================
    # 1. BACA DATASET
    # =========================================================

    dataset_path = Path(
        "data/dataset.txt"
    )

    text = dataset_path.read_text(
        encoding="utf-8"
    )


    # =========================================================
    # 2. BUAT TOKENIZER
    # =========================================================

    tokenizer = CharacterTokenizer(text)


    # =========================================================
    # 3. SIMPAN TOKENIZER
    # =========================================================

    tokenizer.save(
        "models/tokenizer.json"
    )


    # =========================================================
    # 4. TENTUKAN CONTEXT LENGTH
    # =========================================================

    block_size = 32


    # =========================================================
    # 5. BUAT DATASET
    # =========================================================

    dataset = TextDataset(
        text=text,
        tokenizer=tokenizer,
        block_size=block_size
    )


    # =========================================================
    # 6. BUAT DATALOADER
    # =========================================================

    batch_size = 4

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )


    # =========================================================
    # 7. AMBIL SATU BATCH
    # =========================================================

    x_batch, y_batch = next(
        iter(loader)
    )


    # =========================================================
    # 8. INFORMASI DATASET
    # =========================================================

    print("=" * 60)
    print("MICHAEL AI - DATASET PIPELINE")
    print("=" * 60)

    print()

    print(
        "Total karakter :",
        len(text)
    )

    print(
        "Vocabulary     :",
        tokenizer.vocab_size
    )

    print(
        "Block size     :",
        block_size
    )

    print(
        "Total sample   :",
        len(dataset)
    )

    print(
        "Batch size     :",
        batch_size
    )

    print()


    # =========================================================
    # 9. SAMPLE PERTAMA
    # =========================================================

    x, y = dataset[0]

    print("INPUT:")
    print(x)

    print()

    print("TARGET:")
    print(y)

    print()


    # =========================================================
    # 10. DECODE SAMPLE PERTAMA
    # =========================================================

    input_text = tokenizer.decode(
        x.tolist()
    )

    target_text = tokenizer.decode(
        y.tolist()
    )

    print("INPUT TEXT:")
    print(repr(input_text))

    print()

    print("TARGET TEXT:")
    print(repr(target_text))

    print()


    # =========================================================
    # 11. INFORMASI BATCH
    # =========================================================

    print("=" * 60)
    print("BATCH INFORMATION")
    print("=" * 60)

    print()

    print("BATCH INPUT SHAPE:")
    print(x_batch.shape)

    print()

    print("BATCH TARGET SHAPE:")
    print(y_batch.shape)

    print()


    # =========================================================
    # 12. TAMPILKAN BATCH
    # =========================================================

    print("BATCH INPUT:")
    print(x_batch)

    print()

    print("BATCH TARGET:")
    print(y_batch)

    print()


    # =========================================================
    # 13. DECODE SELURUH BATCH
    # =========================================================

    print("=" * 60)
    print("DECODED BATCH")
    print("=" * 60)

    for i in range(batch_size):

        input_text = tokenizer.decode(
            x_batch[i].tolist()
        )

        target_text = tokenizer.decode(
            y_batch[i].tolist()
        )

        print()

        print(
            f"Sample {i + 1}"
        )

        print(
            "Input :",
            repr(input_text)
        )

        print(
            "Target:",
            repr(target_text)
        )


if __name__ == "__main__":
    main()