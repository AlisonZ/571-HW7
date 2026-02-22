import numpy as np
from gensim.models import KeyedVectors

from common import get_average_vector


def get_context_vector(
    sentence_vector: np.ndarray, token_vector: np.ndarray, sentence_length: int
) -> np.ndarray:
    """Compute the context vector for a token by averaging the sentence vector excluding the token's vector.

    Args:
        sentence_vector (np.ndarray): The sentence vector, i.e. the average of all token vectors for every token in the sentence.
        token_vector (np.ndarray): The vector of the specific token.
        sentence_length (int): The number of tokens in the sentence.

    Returns:
        The context vector for the token, i.e. the average of all token vectors excluding this token's vector.
    """
    if sentence_length <= 1:
        print(f"ZXXXXXX {sentence_vector}")
        return np.zeros_like(sentence_vector)
    # TODO (1-4 lines): implement the context vector calculation
    return


def get_global_context_vectors(
    sentence: dict, static_vectors: KeyedVectors
) -> tuple[np.ndarray, ...]:
    """Compute the token context vectors for a given sentence using GloVe vectors.

    Args:
        sentence: A dictionary representing a sentence, with keys "text" and "tokens".
                    Each token is a dictionary with at least keys "text" and "lemma".
        static_vectors: Pre-loaded GloVe vectors.

    Returns:
        A tuple of context vectors, one for each token in the sentence.
    """
    sentence_length = len(sentence["tokens"])
    word_vectors = tuple(
        (
            static_vectors[token["lemma"].lower()]
            if token["lemma"].lower() in static_vectors
            else np.zeros(static_vectors.vector_size)
        )
        for token in sentence["tokens"]
    )
    # TODO (2-4 lines): compute the sentence vector and then the context vectors for each token, returning a tuple of the latter
    # Note that `get_average_vector` has been imported from common.py
    return tuple()  # replace with actual return value
