from datasets import load_dataset
from pathlib import Path

OUTPUT_DIR = Path("data/downloaded/openassistant")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("=" * 60)
print("MICHAEL AI - DOWNLOADING OPENASSISTANT")
print("=" * 60)

print()
print("Downloading OpenAssistant/oasst1...")

dataset = load_dataset(
    "OpenAssistant/oasst1"
)

print()
print("Dataset berhasil dimuat.")

print(dataset)

print()
print("Saving dataset...")

dataset.save_to_disk(
    str(OUTPUT_DIR)
)

print()
print("=" * 60)
print("DOWNLOAD FINISHED")
print("=" * 60)

print()
print("Location:")
print(OUTPUT_DIR)