import pandas as pd
from pathlib import Path
from datasets import load_dataset


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD HUMANEVAL
# ============================================================

def load_humaneval():

    print("\n" + "=" * 60)
    print("Loading HumanEval...")
    print("=" * 60)

    dataset = load_dataset(
        "openai/openai_humaneval",
        split="test"
    )

    df = dataset.to_pandas()

    print("HumanEval loaded successfully!")
    print(f"Number of problems: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Save HumanEval
    output_path = DATA_DIR / "humaneval_dataset.csv"
    df.to_csv(output_path, index=False)

    print(f"HumanEval saved to: {output_path}")

    return df


# ============================================================
# LOAD MBPP
# ============================================================

def load_mbpp():

    print("\n" + "=" * 60)
    print("Loading MBPP...")
    print("=" * 60)

    # IMPORTANT:
    # Use the full Hugging Face repository ID
    dataset = load_dataset(
        "google-research-datasets/mbpp",
        "full",
        split="test"
    )

    df = dataset.to_pandas()

    print("MBPP loaded successfully!")
    print(f"Number of problems: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Save MBPP
    output_path = DATA_DIR / "mbpp_dataset.csv"
    df.to_csv(output_path, index=False)

    print(f"MBPP saved to: {output_path}")

    return df


# ============================================================
# CONVERT DATASETS TO COMMON FORMAT
# ============================================================

def create_common_dataset():

    print("\n" + "=" * 60)
    print("Creating combined programming dataset...")
    print("=" * 60)

    # Load HumanEval
    humaneval_df = load_humaneval()

    # Load MBPP
    mbpp_df = load_mbpp()

    # --------------------------------------------------------
    # HumanEval common columns
    # --------------------------------------------------------

    humaneval_common = pd.DataFrame({
        "dataset": "HumanEval",
        "task_id": humaneval_df["task_id"],
        "prompt": humaneval_df["prompt"],
        "canonical_solution": humaneval_df["canonical_solution"],
        "test": humaneval_df["test"],
        "entry_point": humaneval_df["entry_point"],
        "test_setup_code": "",
        "challenge_test_list": ""
    })

    # --------------------------------------------------------
    # MBPP common columns
    # --------------------------------------------------------

    mbpp_common = pd.DataFrame({
        "dataset": "MBPP",
        "task_id": mbpp_df["task_id"],
        "prompt": mbpp_df["text"],
        "canonical_solution": mbpp_df["code"],
        "test": mbpp_df["test_list"].apply(
            lambda x: "\n".join(x) if isinstance(x, list) else str(x)
        ),
        "entry_point": "",
        "test_setup_code": mbpp_df["test_setup_code"].fillna(""),
        "challenge_test_list": mbpp_df["challenge_test_list"].apply(
            lambda x: "\n".join(x) if isinstance(x, list) else ("" if pd.isna(x) else str(x))
        )
    })

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    combined_df = pd.concat(
        [humaneval_common, mbpp_common],
        ignore_index=True
    )

    # Save combined dataset
    output_path = DATA_DIR / "programming_dataset.csv"

    combined_df.to_csv(
        output_path,
        index=False
    )

    print("\nCombined dataset created successfully!")
    print(f"Total problems: {len(combined_df)}")
    print(f"Columns: {list(combined_df.columns)}")

    print("\nDataset distribution:")
    print(combined_df["dataset"].value_counts())

    print(f"\nSaved to:")
    print(output_path)

    return combined_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    df = create_common_dataset()

    print("\n" + "=" * 60)
    print("FIRST 5 PROBLEMS")
    print("=" * 60)

    print(
        df[
            [
                "dataset",
                "task_id",
                "prompt",
                "entry_point"
            ]
        ].head()
    )
