from pathlib import Path
import hashlib
import random
import re


# ============================================================
# CONFIG
# ============================================================

RAW_DIR = Path("data/raw")

CLEANED_DIR = Path("data/cleaned")

TRAIN_DIR = Path("data/train")

VALIDATION_DIR = Path("data/validation")

TRAIN_RATIO = 0.90

MIN_LINE_LENGTH = 20

RANDOM_SEED = 42


# ============================================================
# DIRECTORY
# ============================================================

def create_directories():

    directories = [
        RAW_DIR,
        CLEANED_DIR,
        TRAIN_DIR,
        VALIDATION_DIR
    ]

    for directory in directories:

        directory.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    # Normalize line endings
    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    # Remove excessive spaces
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Remove excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    # Strip spaces around lines
    lines = []

    for line in text.split("\n"):

        line = line.strip()

        if len(line) >= MIN_LINE_LENGTH:

            lines.append(line)

    return "\n".join(lines)


# ============================================================
# HASH
# ============================================================

def text_hash(text):

    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================
# READ FILES
# ============================================================

def collect_files():

    files = list(
        RAW_DIR.rglob("*.txt")
    )

    print(
        "TXT files found:",
        len(files)
    )

    return files


# ============================================================
# CLEAN DATA
# ============================================================

def clean_dataset(files):

    unique_documents = {}

    total_characters = 0

    duplicate_count = 0

    for file_path in files:

        try:

            text = file_path.read_text(
                encoding="utf-8"
            )

        except UnicodeDecodeError:

            print(
                "Skipped encoding error:",
                file_path
            )

            continue


        cleaned = normalize_text(
            text
        )


        if not cleaned:

            continue


        total_characters += len(
            cleaned
        )


        document_hash = text_hash(
            cleaned
        )


        if document_hash in unique_documents:

            duplicate_count += 1

            continue


        unique_documents[
            document_hash
        ] = cleaned


    print(
        "Unique documents:",
        len(unique_documents)
    )

    print(
        "Duplicates removed:",
        duplicate_count
    )

    print(
        "Characters collected:",
        total_characters
    )

    return list(
        unique_documents.values()
    )


# ============================================================
# SAVE CLEANED DATA
# ============================================================

def save_cleaned(documents):

    output = CLEANED_DIR / "dataset.txt"

    combined = "\n\n".join(
        documents
    )

    output.write_text(
        combined,
        encoding="utf-8"
    )

    print()

    print(
        "Cleaned dataset:",
        output
    )

    print(
        "Cleaned characters:",
        len(combined)
    )

    return combined


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

def split_dataset(text):

    random.seed(
        RANDOM_SEED
    )

    # Split by paragraphs rather than
    # randomly splitting individual characters.

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split(
            "\n\n"
        )
        if paragraph.strip()
    ]


    random.shuffle(
        paragraphs
    )


    split_index = int(
        len(paragraphs)
        * TRAIN_RATIO
    )


    train = paragraphs[
        :split_index
    ]

    validation = paragraphs[
        split_index:
    ]


    train_text = "\n\n".join(
        train
    )

    validation_text = "\n\n".join(
        validation
    )


    TRAIN_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    train_path = (
        TRAIN_DIR
        / "train.txt"
    )

    validation_path = (
        VALIDATION_DIR
        / "validation.txt"
    )


    train_path.write_text(
        train_text,
        encoding="utf-8"
    )

    validation_path.write_text(
        validation_text,
        encoding="utf-8"
    )


    print()

    print(
        "Train paragraphs:",
        len(train)
    )

    print(
        "Validation paragraphs:",
        len(validation)
    )

    print(
        "Train characters:",
        len(train_text)
    )

    print(
        "Validation characters:",
        len(validation_text)
    )


# ============================================================
# STATISTICS
# ============================================================

def print_statistics(text):

    words = re.findall(
        r"\S+",
        text
    )

    lines = text.splitlines()

    print()

    print("=" * 60)

    print(
        "DATASET STATISTICS"
    )

    print("=" * 60)

    print()

    print(
        "Characters:",
        len(text)
    )

    print(
        "Words:",
        len(words)
    )

    print(
        "Lines:",
        len(lines)
    )

    print(
        "Approx MB:",
        round(
            len(text.encode("utf-8"))
            / 1024 / 1024,
            2
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "MICHAEL AI - DATASET ENGINE"
    )

    print("=" * 60)

    print()


    create_directories()


    files = collect_files()


    if not files:

        print()

        print(
            "No .txt files found."
        )

        print()

        print(
            "Put dataset files inside:"
        )

        print(
            "data/raw/"
        )

        return


    documents = clean_dataset(
        files
    )


    if not documents:

        print(
            "No usable documents."
        )

        return


    combined_text = save_cleaned(
        documents
    )


    split_dataset(
        combined_text
    )


    print_statistics(
        combined_text
    )


    print()

    print("=" * 60)

    print(
        "DATASET BUILD FINISHED"
    )

    print("=" * 60)


if __name__ == "__main__":

    main()