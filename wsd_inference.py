import argparse
from collections.abc import Callable
from glob import glob
import json

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from common import load_sense_table, grouped_sense_table, GroupedSenseVectorTable
from contextual_vectors import get_contextual_vectors
from static_vectors import get_global_context_vectors


def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """Compute the cosine similarity between two vectors."""
    if len(vector1.shape) != 1 or len(vector2.shape) != 1:
        raise ValueError("Input vectors must be 1-dimensional.")
    if vector1.shape[0] != vector2.shape[0]:
        raise ValueError("Input vectors must have the same dimensionality.")

    # TODO (1 - 4 lines): Compute cosine similarity between vector1 and vector2
    return


def select_sense(
    context_vector: np.ndarray,
    lemma: str,
    sense_vector_table: GroupedSenseVectorTable,
    mfs_fallback: bool = True,
) -> int:
    """Select the most appropriate sense for a given context vector and lemma.
    If the lemma is not found in the sense vector table, returns -1.

    Args:
        context_vector (np.ndarray): The context vector for the word.
        lemma (str): The lemma of the word.
        sense_vector_table (GroupedSenseVectorTable): The sense vector table (lemma -> sense_id -> vector).
        mfs_fallback (bool): If True, return the most frequent sense when lemma not found.

    Returns:
        int: The selected sense ID. If the lemma is not found, return -1 (or the most frequent sense if mfs_fallback is True).
    """
    # TODO(~5-15 lines): Implement sense selection based on cosine similarity
    # Get all sense vectors for the given lemma (if it's in the table), and return the integer ID with the highest similarity.
    # If the lemma is not found, return either -1, or the most frequent sense if mfs_fallback is True.
    return


def inferences_from_sentence_files(
    file_glob: str,
    context_vectors_function: Callable[[dict], tuple[np.ndarray, ...]],
    sense_vector_table: GroupedSenseVectorTable,
    mfs_fallback: bool = True,
) -> list[dict]:
    """Generate inferences from sentences in files matching the glob pattern."""
    inferences = []
    for file_path in glob(file_glob):
        with open(file_path, "r") as f:
            sentences = json.load(f)
        for sentence in sentences:
            inferences.extend(
                inferences_from_sentence(
                    sentence, context_vectors_function, sense_vector_table, mfs_fallback
                )
            )
    return inferences


def inferences_from_sentence(
    sentence: dict,
    context_vectors_function: Callable[[dict], tuple[np.ndarray, ...]],
    sense_vector_table: GroupedSenseVectorTable,
    mfs_fallback: bool = True,
) -> list[dict]:

    context_vectors = context_vectors_function(sentence)
    inferences = []
    sentence_text = sentence["text"]

    for index, token in enumerate(sentence["tokens"]):
        lemma = token.get("lemma")
        wnsn = token.get("wnsn")
        if lemma is None or wnsn is None:
            continue
        context_vector = context_vectors[index]
        selected_sense = select_sense(
            context_vector, lemma, sense_vector_table, mfs_fallback
        )
        inferences.append(
            {
                "sentence": sentence_text,
                "token": token["text"],
                "lemma": lemma,
                "gold_sense": wnsn,
                "selected_sense": selected_sense,
            }
        )

    return inferences


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="WSD Inference")
    parser.add_argument(
        "--sense_table_path",
        type=str,
        required=True,
        help="Path to the sense table NPZ file",
    )
    parser.add_argument(
        "--semcor_glob", type=str, required=True, help="Path to the SemCor JSON files."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the inference results.",
    )
    parser.add_argument("--mfs_fallback", action="store_true", help="Use MFS fallback.")
    subparsers = parser.add_subparsers(dest="vector_mode", required=True)
    global_parser = subparsers.add_parser("global", help="Use global vectors.")
    global_parser.add_argument(
        "--glove_file",
        type=str,
        default="./data/dolma_300_2024_1.2M.100_combined.txt",
        help="Path to the GloVe file.",
    )
    contextual_parser = subparsers.add_parser(
        "contextual", help="Use contextual vectors."
    )
    # TODO: reset default to  default="/mnt/dropbox/25-26/571W/.cache/distilroberta-base",
    contextual_parser.add_argument(
        "--hf_home",
        type=str,
        default="./distilroberta-base",
        help="Path to the Hugging Face cache directory.",
    )
    contextual_parser.add_argument(
        "--encoder_name",
        type=str,
        default="distilroberta-base",
        help="Name of the contextual encoder.",
    )
    args = parser.parse_args()

    # Load the sense table
    sense_table = load_sense_table(args.sense_table_path)

    # Group the sense table by lemma
    grouped_table = grouped_sense_table(sense_table)

    if args.vector_mode == "global":
        # TODO: uncomment for working version
        # glove_vectors = KeyedVectors.load_word2vec_format(
        #     args.glove_file, binary=False, no_header=True
        # )
        # TODO: remove for real version
        # glove_vectors.save('wsd_glove_vectors.kv')
        glove_vectors = KeyedVectors.load('wsd_glove_vectors.kv')
        inferences = inferences_from_sentence_files(
            args.semcor_glob,
            lambda sentence: get_global_context_vectors(sentence, glove_vectors),
            grouped_table,
            args.mfs_fallback,
        )
    elif args.vector_mode == "contextual":
        # it's not good practice to do these imports in main, but this order of
        # operations is needed in order to have transformers load models from a
        # shared cache directory instead of downloading the model separately for every user
        import os

        os.environ["HF_HOME"] = args.hf_home
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.encoder_name, add_prefix_space=True
        )
        model = AutoModel.from_pretrained(args.encoder_name, output_hidden_states=True)
        inferences = inferences_from_sentence_files(
            args.semcor_glob,
            lambda sentence: get_contextual_vectors(sentence, model, tokenizer),
            grouped_table,
            args.mfs_fallback,
        )
    else:
        raise ValueError(f"Unknown vector mode: {args.vector_mode}")

    inference_data = pd.DataFrame(inferences)
    inference_data.to_csv(args.output_file, index=False)
