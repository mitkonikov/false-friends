import numpy as np

from tqdm import tqdm
from llm import IExtractor
from transliterate import transliterate
from typing import List

def features_and_labels(friend_pairs : List[tuple], llm: IExtractor, lang1 = "mk", lang2 = "sl", lang1_prefix = "Македонски: ", lang2_prefix = "Slovenski: "):
    """Expects a list of friends pairs (lang1, lang2, true/false friend) and returns a X, y with the extracted features

    Args:
        friend_pairs (List[tuple]): List of tuples of the form (string word in lang1, string word in lang2, boolean for is it a true/false friend)
        llm (IExtractor): Class for the feature extraction
        lang1 (str, optional): Language 1 code name. Defaults to "mk".
        lang2 (str, optional): Language 2 code name. Defaults to "sl".
        lang1_prefix (str, optional): Language 1 prefix to add. Defaults to "Македонски: ".
        lang2_prefix (str, optional): Language 2 prefix to add. Defaults to "Slovenski: ".

    Returns:
        tuple: X, y
    """

    print("Computing features...")

    words = {
        lang1: [friend_pair[0] for friend_pair in friend_pairs],
        lang2: [friend_pair[1] for friend_pair in friend_pairs],
    }

    vectors = {
        lang1: np.array([llm.avg_word_embeddings(llm.get_embeddings([lang1_prefix + transliterate(word)], reduced_dim=768)) for word in tqdm(words[lang1])]),
        lang2: np.array([llm.avg_word_embeddings(llm.get_embeddings([lang2_prefix + word], reduced_dim=768)) for word in tqdm(words[lang2])]),
    }

    X = np.hstack((vectors[lang1], vectors[lang2]))
    y = np.array([friend_pair[2] for friend_pair in friend_pairs])
    return X, y
