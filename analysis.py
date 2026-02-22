import argparse
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


def add_mfs_column(inferences: pd.DataFrame, column_name: str = "mfs_sense") -> None:
    """Add a column to the DataFrame representing the MFS baseline."""
    # TODO (~1 line): add column to `inferences` representing what the MFS prediction would be


def calculate_accuracy(
    inferences: pd.DataFrame, gold_column: str, prediction_column: str
) -> float:
    """Calculate accuracy (i.e. percentage of cases where predicted sense equals the gold sense)
    given gold and prediction columns."""
    # TODO (~1-2 lines): Compute accuracy by comparing gold_column and prediction_column
    return


def calculate_precision(inferences: pd.DataFrame, column="selected_sense") -> float:
    """Calculate macro-averaged precision across all senses."""
    return precision_score(
        inferences["gold_sense"],
        inferences[column],
        average="macro",
        zero_division=0,
    )


def calculate_recall(inferences: pd.DataFrame, column="selected_sense") -> float:
    """Calculate macro-averaged recall across all senses."""
    return recall_score(
        inferences["gold_sense"],
        inferences[column],
        average="macro",
        zero_division=0,
    )


def calculate_f1(inferences: pd.DataFrame, column="selected_sense") -> float:
    """Calculate macro-averaged F1 score across all senses."""
    return f1_score(
        inferences["gold_sense"],
        inferences[column],
        average="macro",
        zero_division=0,
    )


if __name__ == "__main__":

    # Parse command-line argument for inference CSV file
    parser = argparse.ArgumentParser(description="WSD Analysis")
    parser.add_argument(
        "--inference_file",
        type=str,
        required=True,
        help="Path to the inference CSV file.",
    )
    args = parser.parse_args()

    # Load the inference data
    inference_data = pd.read_csv(args.inference_file)

    print(f"------ {args.inference_file} ------")
    print("Acc \t P \t R \t F1")
    accuracy = calculate_accuracy(inference_data, "gold_sense", "selected_sense")
    precision = calculate_precision(inference_data)
    recall = calculate_recall(inference_data)
    f1 = calculate_f1(inference_data)
    print(f"{accuracy:.2f} \t {precision:.2f} \t {recall:.2f} \t {f1:.2f}")

    print("\n------ MFS Baseline ------")
    # add MFS as a column to the dataframe, then recompute metrics
    add_mfs_column(inference_data)
    accuracy = calculate_accuracy(inference_data, "gold_sense", "mfs_sense")
    precision = calculate_precision(inference_data, column="mfs_sense")
    recall = calculate_recall(inference_data, column="mfs_sense")
    f1 = calculate_f1(inference_data, column="mfs_sense")
    print("Acc \t P \t R \t F1")
    print(f"{accuracy:.2f} \t {precision:.2f} \t {recall:.2f} \t {f1:.2f}")
