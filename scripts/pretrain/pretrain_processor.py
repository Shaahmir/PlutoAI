import pyarrow.parquet as pq
from pathlib import Path
from config import CONFIG
from tqdm import tqdm

REQUIRED_COULMNS = [
    "id",
    "text",
    "score",
    "language_score",
    "token_count"
]

def validate_schema(file_path: Path):

    parquet_file = pq.ParquetFile(file_path)
    columns = set(parquet_file.schema_arrow.names)
    missing_columns = set(REQUIRED_COULMNS) - columns

    if missing_columns:
        raise ValueError(f"{file_path.name} is missing {missing_columns} column(s)!")
    
def iter_pretrain_rows(file_path: Path, split: str | None = None, manifest: dict[str, str] | None = None):

    validate_schema(file_path)
    parquet_file = pq.ParquetFile(file_path)
    
    for batch in parquet_file.iter_batches(batch_size = 8192, columns = REQUIRED_COULMNS):

        for row in batch.to_pylist():

            document_id = row.get("id")
            score = row.get("score")
            token_count = row.get("token_count")
            language_score = row.get("language_score")
            text = row.get("text")

            if not document_id or score is None or token_count is None or language_score is None or text is None:
                continue

            if score < CONFIG.PRETRAIN_MIN_SCORE or token_count < CONFIG.PRETRAIN_MIN_TOKEN_COUNT or language_score < CONFIG.PRETRAIN_MIN_LANG_SCORE:
                continue

            document_id = str(document_id)

            if split is not None:

                if manifest is None:
                    raise RuntimeError("Manifest can not be None wehn split is specified!")

                if manifest.get(document_id) != split:
                    continue

            yield {
                "id": document_id,
                "text": text,
                "score": float(score),
                "token_count": int(token_count),
                "language_score": float(language_score)
            }

def iter_pretrain_text(split: str = "train"):

    manifest = load_pretrain_manifest()
    file_paths = sorted((CONFIG.RAW_PRETRAIN_DIR).glob("*.parquet"))

    for file_path in file_paths:

        tqdm.write(f"Reading: {file_path.name}")

        for row in iter_pretrain_rows(file_path, split, manifest):
            yield row["text"]

def load_pretrain_manifest():

    table = pq.read_table(
        CONFIG.PROCESSED_DATA / "pretrain_manifest.parquet",
        columns = ["id", "split"]
    )

    ids = table["id"].to_pylist()
    splits = table["split"].to_pylist()

    return dict(
        zip(ids, splits)
    )