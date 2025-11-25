import json
from pathlib import Path

val_path = Path("data/processed/funsd_layoutlmv3_val.jsonl")

# Get all unique doc_ids in validation
val_doc_ids = set()
with open(val_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        val_doc_ids.add(data['doc_id'])

print("Documents in validation set:")
for doc_id in sorted(val_doc_ids):
    print(f"  {doc_id}")

print(f"\nTotal: {len(val_doc_ids)} documents")

# Check which examples you have
example_files = ["87528321", "87528380", "87594142", "89856243", "91814768", "92380595", "93106788"]

print("\n" + "="*50)
print("Which of your examples are in validation?")
print("="*50)

for ex in example_files:
    if ex in val_doc_ids or any(ex in doc for doc in val_doc_ids):
        matches = [doc for doc in val_doc_ids if ex in doc]
        print(f"✅ {ex}.png → Matches: {matches}")
    else:
        print(f"❌ {ex}.png → NOT in validation set")