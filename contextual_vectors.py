import numpy as np
import torch.nn as nn
from torch import Tensor
from transformers import BatchEncoding
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


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
    # TODO (1-4 lines): extract from `token_vectors` the vectors corresponding to this word's tokens,
    # and return their average as a numpy array
    # Note: you can use the `tensor_to_numpy` helper defined above to convert a tensor to a numpy array
    return


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
    # TODO (1-3 lines): get the token vectors from `last_layer` (i.e. shape (sequence_length, hidden_size)
    # and return a tuple of word vectors, using `get_word_vector` defined above
    # Note: since we only pass in one sentence, batch_size = 1
    return tuple()  # replace with actual return value
