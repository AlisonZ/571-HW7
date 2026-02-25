import numpy as np
import torch.nn as nn
from torch import Tensor
from transformers import BatchEncoding
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from common import get_average_vector


def tensor_to_numpy(tensor: Tensor) -> np.ndarray:
    """Helper to convert a PyTorch tensor to a NumPy array."""
    return tensor.detach().numpy()


def get_word_vector(
    inputs: BatchEncoding, token_vectors: Tensor, word_index: int
) -> np.ndarray:
    """Get the vector for a specific word by averaging its token vectors.
    This is necessary because some words may be tokenized into multiple subword tokens.

    Args:
        inputs (BatchEncoding): The tokenized inputs from the tokenizer.
        token_vectors (Tensor): The tensor of token vectors from the model's output.  Shape: (sequence_length, hidden_size).
        word_index (int): The index of the word in the original input.
    """
    # token_indices will be a `TokenSpan` object, containing fields `start` and `end`
    # see https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/tokenizer#transformers.BatchEncoding.word_to_tokens
    # and https://huggingface.co/docs/transformers/v4.57.1/en/internal/tokenization_utils#transformers.TokenSpan
    token_indices = inputs.word_to_tokens(word_index)
    if token_indices is None:
        raise ValueError(f"Word index {word_index} not found in inputs.")

    word_vectors = token_vectors[token_indices.start:token_indices.end]
    word_vectors_avg = get_average_vector(tensor_to_numpy(word_vectors))

    return word_vectors_avg


def get_contextual_vectors(
    sentence: dict, model: nn.Module, tokenizer: PreTrainedTokenizerBase
) -> tuple[np.ndarray, ...]:
    """Compute the contextual vectors for a given sentence using a transformer model.

    Args:
        sentence: A dictionary representing a sentence, with keys "text" and "tokens".
        model: A pre-loaded transformer model from Hugging Face.
        tokenizer: A pre-loaded tokenizer corresponding to the model.

    Returns:
        A tuple of contextual vectors, one for each _word_ in the sentence.
    """
    words = [word["text"] for word in sentence["tokens"]]
    inputs = tokenizer(
        words,
        return_tensors="pt",
        is_split_into_words=True,
    )
    outputs = model(**inputs)
    # hidden states = tuple, one for each layer
    hidden_states = outputs.hidden_states
    # last_layer: (batch_size, sequence_length, hidden_size)
    last_layer = hidden_states[-1]
    token_vectors = last_layer[0]
    word_vectors = []
    for i in range(len(words)):
        word_vector = get_word_vector(inputs, token_vectors, i)
        word_vectors.append(word_vector)

    return tuple(word_vectors) 
