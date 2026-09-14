import pyarrow.dataset as ds
import pyarrow.compute as pc
from pathlib import Path

files = sorted(Path(__file__).resolve().parent.glob("*.parquet"))

total_rows = 0
total_tokens = 0

for file in files:
    
    dataset = ds.dataset(file, format = "parquet")

    filter_expr = (
        (ds.field("score") >= 2.5) &
        (ds.field("token_count") >= 128) &
        (ds.field("language_score") >= 0.85)
    )

    table = dataset.to_table(filter = filter_expr, columns = ["token_count"])
    
    kept_rows = table.num_rows
    file_tokens = pc.max(table["token_count"]).as_py() or 0
    
    total_rows += kept_rows
    total_tokens += file_tokens
    
    print(f"{file.name}: max tokens = {file_tokens:,}")

print("\n" + "="*40)
print(f"Total Filtered Rows:   {total_rows:,}")
print(f"Total Filtered Tokens: {total_tokens:,}")