from pathlib import Path


class CharacterTokenizer:

    def __init__(self, text):

        # Semua karakter unik dalam dataset
        self.vocab = sorted(set(text))

        # Karakter -> ID
        self.char_to_id = {
            char: idx
            for idx, char in enumerate(self.vocab)
        }

        # ID -> karakter
        self.id_to_char = {
            idx: char
            for idx, char in enumerate(self.vocab)
        }

        self.vocab_size = len(self.vocab)


    def encode(self, text):
        """
        Mengubah teks menjadi list angka.
        """

        return [
            self.char_to_id[char]
            for char in text
        ]


    def decode(self, ids):
        """
        Mengubah list angka menjadi teks.
        """

        return "".join(
            self.id_to_char[idx]
            for idx in ids
        )


    def save(self, path):
        """
        Menyimpan vocabulary.
        """

        import json

        data = {
            "vocab": self.vocab
        }

        Path(path).write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )


    @classmethod
    def load(cls, path):
        """
        Memuat vocabulary tokenizer.
        """

        import json

        data = json.loads(
            Path(path).read_text(
                encoding="utf-8"
            )
        )

        tokenizer = cls("".join(data["vocab"]))

        return tokenizer