from config import CONFIG
from data.dataset import PackedTokenDataset

def verify_split(split: str, num_samples = 1000):

    split_dir = CONFIG.PROCESSED_DATA / split / CONFIG.DATASET_MODE
    dataset = PackedTokenDataset(split_dir)

    print("=" * 50)
    print(f"{split.upper()} DATASET")

    print(f"Sequences: {len(dataset)}")
    print(f"Context length: {CONFIG.CONTEXT_LENGTH}")

    sample = dataset[0]

    print(f"Input shape: {tuple(sample['input_ids'].shape)}")
    print(f"Label shape: {tuple(sample['labels'].shape)}")

    print(f"Input dtype: {sample['input_ids'].dtype}")
    print(f"Label dtype: {sample['labels'].dtype}")

    sample_count = min(num_samples, len(dataset))

    ignored_counter = 0
    trained_counter = 0
    fully_ignored = 0
    fully_trained = 0
    partially_masked = 0

    for index in range(sample_count):

        item = dataset[index]

        if "loss_mask" in item:
            ignored = (item["loss_mask"] == 0).sum().item()
            trained = (item["loss_mask"] == 1).sum().item()

        else:
            ignored = 0
            trained = (item["labels"].numel())

        ignored_counter += ignored
        trained_counter += trained

        if ignored == 0:
            fully_trained += 1

        elif trained == 0:
            fully_ignored += 1

        else:
            partially_masked += 1

    total_tokens = ignored_counter + trained_counter

    print(f"Samples inspected: {sample_count:,}")

    print(f"Ignored tokens: {ignored_counter:,} {ignored_counter / total_tokens:.2f}")
    print(f"Trained tokens: {trained_counter:,} {trained_counter / total_tokens:.2f}")

    if CONFIG.DATASET_MODE == "sft":
        print(f"Fully trained sequences: {fully_trained:,}")
        print(f"Partially masked sequences: {partially_masked:,}")
        print(f"Fully ignored sequences: {fully_ignored:,}")

    if fully_ignored > 0:
        raise RuntimeError("Found sequences containing no training tokens!")

def main():
    verify_split("train")
    verify_split("valid")

if __name__ == "__main__":
    main()