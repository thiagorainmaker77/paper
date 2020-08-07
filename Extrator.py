
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from collections import defaultdict
from sklearn.preprocessing import FunctionTransformer
from logging import WARNING, Formatter, StreamHandler, getLogger
import os
from typing import Iterable, List, Union

import gensim.downloader as api
import numpy as np
from gensim.models.keyedvectors import Word2VecKeyedVectors
from sklearn.base import BaseEstimator, TransformerMixin


"""#Biblioteca de extração"""

OOV_TAG = "<oov>"

DEFAULT_PRETRAINED_EMBEDDINGS = {
    "conceptnet-numberbatch-17-06-300" : "conceptnet-numberbatch-17-06-300",
    "fasttext-wiki-news-subwords-300":"fasttext-wiki-news-subwords-300",
    "glove-twitter-100": "glove-twitter-100",
    "glove-twitter-200":"glove-twitter-200",
    "glove-twitter-25":"glove-twitter-25",
    "glove-twitter-50":"glove-twitter-50",
    "glove-wiki-gigaword-100":"glove-wiki-gigaword-100",
    "glove-wiki-gigaword-200":"glove-wiki-gigaword-200",
    "glove-wiki-gigaword-300":"glove-wiki-gigaword-300",
    "glove-wiki-gigaword-50":"glove-wiki-gigaword-50",
    "word2vec-google-news-300":"word2vec-google-news-300",
    "word2vec-ruscorpora-300":"word2vec-ruscorpora-300",
}





LOG_FORMAT = "%(asctime)s [%(levelname)s]: %(message)s in %(pathname)s:%(lineno)d"

DEFAULT_LOG_LEVEL = WARNING


DEFAULT_HANDLER = StreamHandler()
DEFAULT_HANDLER.setFormatter(Formatter(LOG_FORMAT))

package_logger = getLogger("zeugma")
package_logger.setLevel(DEFAULT_LOG_LEVEL)
package_logger.addHandler(DEFAULT_HANDLER)




class Gerar_extrator(BaseEstimator, TransformerMixin):
    model: Word2VecKeyedVectors
    aggregation: str

    def __init__(self, model: str = "glove", aggregation: str = "average"):
        if aggregation not in {"average", "sum", "minmax"}:
            raise ValueError(
                f"Unknown embeddings aggregation mode: {aggregation}, the available "
                "ones are: average, sum, or minmax."
            )
        if isinstance(model, str):

            model = model.lower()
            if model in DEFAULT_PRETRAINED_EMBEDDINGS.keys():
                model_gensim_name = DEFAULT_PRETRAINED_EMBEDDINGS[model]
                self.model = api.load(model_gensim_name)
            elif model in api.info()["models"].keys():
                self.model = api.load(model)  # pragma: no cover
            elif os.path.exists(model):
                self.model = Word2VecKeyedVectors.load(model)
                if not isinstance(self.model, Word2VecKeyedVectors):
                    raise TypeError(
                        "The input model should be a Word2VecKeyedVectors object but "
                        f"it is a {type(self.model)} object."
                    )
            else:
                raise KeyError(
                    f"Unknown pre-trained model name: {model}. Available models are"
                    + ", ".join(api.info()["models"].keys())
                )
        elif isinstance(model, Word2VecKeyedVectors):
            self.model = model
        else:
            raise TypeError(
                "ERRO"
            )
        self.aggregation = aggregation
        self.embedding_dimension = self.model.vector_size
        if self.aggregation == "minmax":
            self.embedding_dimension *= 2

    def transform_sentence(self, text: Union[Iterable, str]) -> np.array:

        def preprocess_text(raw_text: Union[Iterable, str]) -> List[str]:
            if not isinstance(raw_text, list):
                if not isinstance(raw_text, str):
                    raise TypeError(
                        f"ERRO {type(raw_text)}"
                    )
                raw_tokens = raw_text.split()
            return list(filter(lambda x: x in self.model.vocab, raw_tokens))

        tokens = preprocess_text(text)

        if not tokens:
            return np.zeros(self.embedding_dimension, dtype=np.float32)

        if self.aggregation == "average":
            return np.mean(self.model[tokens], axis=0)
        elif self.aggregation == "sum":
            return np.sum(self.model[tokens], axis=0)
        elif self.aggregation == "minmax":
            maxi = np.max(self.model[tokens], axis=0)
            mini = np.min(self.model[tokens], axis=0)
            return np.append(mini, maxi)

    def fit(self, x: Iterable[Iterable], y: Iterable = None) -> BaseEstimator:
        return self

    def transform(self, texts: Iterable[str]) -> Iterable[Iterable]:
        return np.array([self.transform_sentence(t) for t in texts])



class TextsToSequences(Tokenizer, BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, texts, y=None):
        self.fit_on_texts(texts)
        return self

    def transform(self, texts, y=None):
        return np.array(self.texts_to_sequences(texts))


class Padder(BaseEstimator, TransformerMixin):
    def __init__(self, max_length=500):
        self.max_length = max_length
        self.max_index = None

    def fit(self, X, y=None):
        self.max_index = pad_sequences(X, maxlen=self.max_length).max()
        return self

    def transform(self, X, y=None):
        X = pad_sequences(X, maxlen=self.max_length)
        X[X > self.max_index] = 0
        return X


if __name__ == "__main__":
    import doctest

    doctest.testmod()




class RareWordsTagger(BaseEstimator, TransformerMixin):
    """ Replace rare words with a token in a corpus (list of strings) """

    def __init__(self, min_count, oov_tag=OOV_TAG):
        self.min_count = min_count
        self.oov_tag = oov_tag
        self.frequencies = defaultdict(int)
        self.kept_words = None

    def fit(self, texts, y=None):
        all_tokens = (token for t in texts for token in t.split())
        for w in all_tokens:
            self.frequencies[w] += 1
        return self

    def transform(self, texts):
        texts = [
            " ".join((w if w in self.kept_words else self.oov_tag for w in t.split()))
            for t in texts
        ]
        return texts


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class Namer(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return {self.key: X}


class TextStats(FunctionTransformer):
    def __init__(self):
        def extract_stats(corpus):
            return [
                {"length": len(text), "num_sentences": text.count(".")}
                for text in corpus
            ]

        super().__init__(extract_stats, validate=False)