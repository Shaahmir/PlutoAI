import hashlib
import pyarrow as pa
import pyarrow.parquet as pq

from pathlib import Path
from config import CONFIG

MANIFEST_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("source_file", pa.string()),
    ("row_index", pa.int64()),
    ("split", pa.string())
])


def assign_split(document_id: str):

    digest = hashlib.sha256(
        document_id.encode("utf-8")
    ).digest()

    bucket = int.from_bytes(digest[:8], byteorder = "big", signed = False) / 2 ** 64

    if bucket < CONFIG.VALIDATION_SPLIT:
        return "valid"

    return "train"

def validate_schema(file_path: Path):
    parquet_file = pq.ParquetFile(file_path)
    columns = set(parquet_file.schema_arrow.names)

    required = {
        "id",
        "text",
        "score",
        "token_count",
        "language_score"
    }

    missing = required - columns

    if missing:
        raise RuntimeError("Some required columns are missing!")

def process_file(file_path: Path, writer: pq.ParquetWriter | None):

    validate_schema(file_path)
    parquet_file = pq.ParquetFile(file_path)

    train_count = 0
    valid_count = 0
    skipped_count = 0
    global_row_index = 0

    for batch in parquet_file.iter_batches(batch_size = 8192, columns = ["id", "score", "token_count", "language_score"]):

        rows = batch.to_pylist()
        manifest_rows = []

        for offset, row in enumerate(rows):

            document_id = row.get("id")
            score = row.get("score")
            token_count = row.get("token_count")
            language_score = row.get("language_score")

            if not document_id or score is None or token_count is None or language_score is None:
                skipped_count += 1
                continue

            if score < CONFIG.PRETRAIN_MIN_SCORE or token_count < CONFIG.PRETRAIN_MIN_TOKEN_COUNT or language_score < CONFIG.PRETRAIN_MIN_LANG_SCORE:
                skipped_count += 1
                continue

            document_id = str(document_id)
            split = assign_split(document_id)

            manifest_rows.append({
                "id": document_id,
                "source_file": file_path.name,
                "row_index": global_row_index + offset,
                "split": split
            })

            if split == "train":
                train_count += 1
            else:
                valid_count += 1

        if manifest_rows:
            table = pa.Table.from_pylist(manifest_rows, schema = MANIFEST_SCHEMA)
            writer.write_table(table)

        global_row_index += len(rows)

    return (
        train_count,
        valid_count,
        skipped_count
    )

def validate_manifest(manifest_path: Path):

    table = pq.read_table(manifest_path, columns = ["id", "split"])

    ids = table["id"].to_pylist()
    splits = table.column("split").to_pylist()

    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate IDs Found!")

    train_count = sum(split == "train" for split in splits)
    valid_count = sum(split == "valid" for split in splits)
    total = len(splits)

    actual_validation_ratio = valid_count / total if total else 0.0

    print("=" * 50)
    print("MANIFEST SUMMARY:")

    print(f"Total Documents: {total:,}")
    print(f"Train Documents: {train_count:,}")
    print(f"Valid Documents: {valid_count:,}")
    print(f"Train Ratio: {train_count / total:.4%}")
    print(f"Valid Ratio: {actual_validation_ratio:.4%}")
    print(f"Target Valid Ratio: {CONFIG.VALIDATION_SPLIT:.4%}")
    print()
    
    if total == 0:
        raise ValueError("No Document detected!")

    if abs(actual_validation_ratio - CONFIG.VALIDATION_SPLIT) > 0.001:
        raise ValueError("Validation ratio deviated unexpectedly!")

def main():
    output_path = CONFIG.PROCESSED_DATA / "pretrain_manifest.parquet"

    if output_path.exists():
        output_path.unlink()

    writer = None
    total_train = 0
    total_valid = 0
    total_skipped = 0

    file_paths = sorted((CONFIG.RAW_PRETRAIN_DIR).glob("*.parquet"))

    try:
        writer = pq.ParquetWriter(
            where = output_path,
            schema = MANIFEST_SCHEMA,
            compression = "zstd",
            use_dictionary = True
        )

        for file_path in file_paths:

            print(f"Creating manifest {file_path.name}")

            train_count, valid_count, skipped = process_file(
                file_path,
                writer
            )

            total_train += train_count
            total_valid += valid_count
            total_skipped += skipped

            print(f"Train: {train_count:,}")
            print(f"Valid: {valid_count:,}")
            print(f"Skipped: {skipped:,}")

    finally:
        if writer is not None:
            writer.close()

    validate_manifest(output_path)

    print(f"Train Documents {total_train:,}")
    print(f"Valid Documents {total_valid:,}")
    print(f"Skipped Documents {total_skipped:,}")

if __name__ == "__main__":
    main()