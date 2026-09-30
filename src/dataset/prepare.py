import argparse
import json
import pandas as pd
from pathlib import Path

def prepare_jsonl(input_path, output_path):
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append({
                "problem_id": item.get("task_id", f"task_{len(rows)+1}"),
                "prompt": item.get("prompt", ""),
                "tests": item.get("test", ""),
                "required_features": json.dumps([])
            })
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Wrote {len(rows)} rows to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl")
    parser.add_argument("output_csv")
    args = parser.parse_args()
    prepare_jsonl(args.input_jsonl, args.output_csv)
