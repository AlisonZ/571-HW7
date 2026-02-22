from collections.abc import Sequence
import numpy as np

# (lemma-sense_id) -> vector
# this format enables easy saving and loading via numpy
SenseVectorTable = dict[str, np.ndarray]
# lemma -> (sense_id -> vector)
# this format enables quick lookup of all senses of a given lemma
GroupedSenseVectorTable = dict[str, dict[int, np.ndarray]]


def get_average_vector(token_vectors: Sequence[np.ndarray]) -> np.ndarray:
    return np.mean(token_vectors, axis=0)


def save_sense_table(table: SenseVectorTable, output_file: str) -> None:
    """Save the sense vector table to a plaintext file."""
    with open(output_file, "w") as f:
        for key, vector in table.items():
            vector_str = "\t".join(f"{x:.6f}" for x in vector)
            f.write(f"{key}\t{vector_str}\n")


def load_sense_table(input_file: str) -> SenseVectorTable:
    """Load the sense vector table from a plaintext file."""
    table: SenseVectorTable = {}
    with open(input_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            key = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            table[key] = vector
    return table


def grouped_sense_table(table: SenseVectorTable) -> GroupedSenseVectorTable:
    """Group the sense vector table by lemma for more efficient lookup.
    Takes a SenseVectorTable (lemma-sense_id -> vector) and returns a
    GroupedSenseVectorTable (lemma -> sense_id -> vector).
    """
    grouped_table: GroupedSenseVectorTable = {}
    for key, vector in table.items():
        lemma, sense_id_str = key.rsplit("-", 1)
        sense_id = int(sense_id_str)
        if lemma not in grouped_table:
            grouped_table[lemma] = {}
        grouped_table[lemma][sense_id] = vector
    return grouped_table
