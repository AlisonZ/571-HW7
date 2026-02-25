import argparse
from collections import Counter
from collections.abc import Callable
from glob import glob
import json

import numpy as np
from gensim.models import KeyedVectors

from common import save_sense_table, SenseVectorTable
from contextual_vectors import get_contextual_vectors
from static_vectors import get_global_context_vectors


def senses_from_sentence_files(
    file_glob: str,
    context_vectors_function: Callable[[dict], tuple[np.ndarray, ...]],
) -> SenseVectorTable:
    table: SenseVectorTable = {}
    sense_counts: Counter = Counter()
    # Iterate over each file matching the glob pattern, adding senses to the table
    for file_path in glob(file_glob):
        with open(file_path, "r") as f:
            sentences = json.load(f)
        for sentence in sentences:
            add_senses_from_sentence(
                sentence, context_vectors_function, table, sense_counts
            )
    # divide each sense vector by its corresponding count in order to get an average
    for key, vector in table.items():
        table[key] = vector / sense_counts[key]
    return table


def add_senses_from_sentence(
    sentence: dict,
    context_vectors_function: Callable[[dict], tuple[np.ndarray, ...]],
    table: SenseVectorTable,
    sense_counts: Counter,
) -> None:
    """Add sense vectors from a single sentence to the provided table.

    Args:
        sentence: A dictionary representing a sentence, with keys "text" and "tokens".
                  Each token is a dictionary with possible keys "text", "lemma", and "wnsn".
        glove_vectors: Pre-loaded GloVe vectors.
        table: A dictionary mapping (lemma, sense_id) to their corresponding vectors.
        sense_counts: A Counter tracking the number of occurrences of each lemma-sense.
    """
    context_vectors = context_vectors_function(sentence)
    # print(f'HIII {context_vectors}')
    # for index, token in enumerate(sentence["tokens"]):
    #     lemma = token.get("lemma")
    #     wnsn = token.get("wnsn")
    #     if lemma is None or wnsn is None:
    #         continue
    #     context_vector = context_vectors[index]
    #     key = f"{lemma}-{wnsn}"
    #     if key not in table:
    #         table[key] = context_vector
    #     else:
    #         table[key] += context_vector
    #     sense_counts[key] += 1


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Generate sense representations from SemCor and GloVe vectors."
    )
    parser.add_argument(
        "--semcor_glob", type=str, required=True, help="Path to the SemCor JSON files."
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to the output file."
    )
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

    contextual_parser.add_argument(
        "--hf_home",
        type=str,
        default="/mnt/dropbox/25-26/571W/.cache/distilroberta-base",
        help="Path to the Hugging Face cache directory.",
    )
    contextual_parser.add_argument(
        "--encoder_name",
        type=str,
        default="distilroberta-base",
        help="Name of the contextual encoder.",
    )
    args = parser.parse_args()

    if args.vector_mode == "global":
        # TODO: uncomment for working version
        # glove_vectors = KeyedVectors.load_word2vec_format(
        #     args.glove_file, binary=False, no_header=True
        # )
        # TODO: Remove for real version. This is for testikng
        # glove_vectors.save('glove_vectors.kv')
        glove_vectors = KeyedVectors.load('glove_vectors.kv')
        sense_table = senses_from_sentence_files(
            args.semcor_glob,
            lambda sentence: get_global_context_vectors(sentence, glove_vectors),
        )
        save_sense_table(sense_table, args.output_file)

    elif args.vector_mode == "contextual":
        print(f'CONTEXT {args.output_file}')
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
        sense_table = senses_from_sentence_files(
            args.semcor_glob,
            lambda sentence: get_contextual_vectors(sentence, model, tokenizer),
        )
        # save_sense_table(sense_table, args.output_file)
