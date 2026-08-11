from pathlib import Path

from datasets import load_from_disk


# ============================================================
# CONFIG
# ============================================================

SOURCE_DIR = Path(
    "data/downloaded/openassistant"
)

OUTPUT_DIR = Path(
    "data/processed"
)

TRAIN_OUTPUT = (
    OUTPUT_DIR / "openassistant_train.txt"
)

VALIDATION_OUTPUT = (
    OUTPUT_DIR / "openassistant_validation.txt"
)

MIN_TEXT_LENGTH = 10

# Bahasa yang kita izinkan
ALLOWED_LANGUAGES = {
    "id",
    "en"
}


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    if not isinstance(text, str):
        return ""

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    lines = []

    for line in text.split("\n"):

        line = line.strip()

        if line:
            lines.append(line)

    text = "\n".join(lines)

    return text.strip()


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset_local():

    print("=" * 60)

    print(
        "MICHAEL AI - OPENASSISTANT CONVERTER"
    )

    print("=" * 60)

    print()

    print(
        "Loading:",
        SOURCE_DIR
    )

    dataset = load_from_disk(
        str(SOURCE_DIR)
    )

    print()

    print(dataset)

    return dataset


# ============================================================
# BUILD MESSAGE INDEX
# ============================================================

def build_message_index(dataset):

    index = {}

    total = 0

    for split_name in dataset:

        split = dataset[split_name]

        for row in split:

            message_id = row["message_id"]

            if not message_id:
                continue

            index[message_id] = row

            total += 1

    print()

    print(
        "Messages indexed:",
        total
    )

    return index


# ============================================================
# VALID MESSAGE
# ============================================================

def valid_message(row):

    if row is None:
        return False

    text = clean_text(
        row.get("text", "")
    )

    if len(text) < MIN_TEXT_LENGTH:
        return False

    if row.get("deleted", False):
        return False

    language = row.get(
        "lang",
        None
    )

    if language not in ALLOWED_LANGUAGES:
        return False

    role = row.get(
        "role",
        None
    )

    if role not in {
        "prompter",
        "assistant"
    }:
        return False

    return True


# ============================================================
# NORMALIZE ROLE
# ============================================================

def role_name(role):

    if role == "prompter":
        return "USER"

    if role == "assistant":
        return "ASSISTANT"

    return None


# ============================================================
# BUILD CONVERSATION
# ============================================================

def build_conversation(
    row,
    message_index
):

    conversation = []

    current = row

    visited = set()

    while current is not None:

        message_id = current.get(
            "message_id"
        )

        if not message_id:
            break

        if message_id in visited:
            break

        visited.add(
            message_id
        )

        if not valid_message(current):
            break

        role = role_name(
            current.get("role")
        )

        text = clean_text(
            current.get("text", "")
        )

        if role is None:
            break

        conversation.append(
            (
                role,
                text
            )
        )

        parent_id = current.get(
            "parent_id"
        )

        if not parent_id:
            break

        current = message_index.get(
            parent_id
        )


    conversation.reverse()

    return conversation


# ============================================================
# CONVERSATION VALIDATION
# ============================================================

def valid_conversation(
    conversation
):

    if len(conversation) < 2:
        return False

    has_user = any(
        role == "USER"
        for role, _ in conversation
    )

    has_assistant = any(
        role == "ASSISTANT"
        for role, _ in conversation
    )

    if not has_user:
        return False

    if not has_assistant:
        return False

    return True


# ============================================================
# FORMAT CONVERSATION
# ============================================================

def format_conversation(
    conversation
):

    parts = []

    for role, text in conversation:

        parts.append(
            f"{role}: {text}"
        )

    return "\n".join(parts)


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate(
    conversations
):

    seen = set()

    result = []

    for conversation in conversations:

        key = conversation.strip()

        if not key:
            continue

        if key in seen:
            continue

        seen.add(key)

        result.append(key)

    return result


# ============================================================
# PROCESS SPLIT
# ============================================================

def process_split(
    split,
    message_index
):

    results = []

    processed = 0

    valid = 0

    for row in split:

        processed += 1

        conversation = build_conversation(
            row,
            message_index
        )

        if not valid_conversation(
            conversation
        ):
            continue

        formatted = format_conversation(
            conversation
        )

        if len(formatted) < 30:
            continue

        results.append(
            formatted
        )

        valid += 1

    print()

    print(
        "Rows processed:",
        processed
    )

    print(
        "Valid conversations:",
        valid
    )

    return deduplicate(
        results
    )


# ============================================================
# SAVE
# ============================================================

def save_text(
    path,
    conversations
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    text = "\n\n".join(
        conversations
    )

    path.write_text(
        text,
        encoding="utf-8"
    )

    print()

    print(
        "Saved:",
        path
    )

    print(
        "Characters:",
        len(text)
    )

    print(
        "Conversations:",
        len(conversations)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    dataset = load_dataset_local()

    message_index = build_message_index(
        dataset
    )


    # ========================================================
    # TRAIN
    # ========================================================

    print()

    print("=" * 60)

    print(
        "PROCESSING TRAIN"
    )

    print("=" * 60)

    train_conversations = process_split(
        dataset["train"],
        message_index
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    print()

    print("=" * 60)

    print(
        "PROCESSING VALIDATION"
    )

    print("=" * 60)

    validation_conversations = process_split(
        dataset["validation"],
        message_index
    )


    # ========================================================
    # SAVE
    # ========================================================

    save_text(
        TRAIN_OUTPUT,
        train_conversations
    )

    save_text(
        VALIDATION_OUTPUT,
        validation_conversations
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print()

    print("=" * 60)

    print(
        "CONVERSION FINISHED"
    )

    print("=" * 60)

    print()

    print(
        "Train:",
        TRAIN_OUTPUT
    )

    print(
        "Validation:",
        VALIDATION_OUTPUT
    )


if __name__ == "__main__":

    main()