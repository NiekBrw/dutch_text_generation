from __future__ import annotations

import re
import logging
import multiprocessing
LOGGER = logging.getLogger(__name__)

from typing import Sequence, Optional, Callable, List, Any
from instancelib.feature_extraction.base import BaseVectorizer

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SBERT_AVAILABLE = False
else:
    SBERT_AVAILABLE = True

try:
    import spacy
except ImportError:
    SPACY_AVAILABLE = False
else:
    SPACY_AVAILABLE = True

def _check_sbert_imports():
    if not SBERT_AVAILABLE:
        raise ImportError("Install sentence-bert by `pip install sentence-transformers`")


def _check_spacy_imports():
    if not SPACY_AVAILABLE:
        raise ImportError("Install spacy by `pip install spacy`")


import numpy as np

np.random.seed(0)

class PretrainedSentenceBERTVectorizer(BaseVectorizer[str]):
    name = "SentenceBERT"
    def __init__(
        self,
        bert_model_name: str = "",
        spacy_model_name: str = "",
        save_path: str = None
    ) -> None:
        super().__init__()
        self.bert_model_name = bert_model_name
        self.spacy_model_name = spacy_model_name
        self._nlp = None
        self._bert = None
        self._cross_encoder = None
        self._fitted = True
        self.num_workers = multiprocessing.cpu_count()
        self.taboo_fields = {"_bert": None, "_nlp": None, "_cross_encoder": None}
    
    @property
    def bert(self) -> "SentenceTransformer":
        _check_sbert_imports()
        if self._bert is None:
            self._bert = SentenceTransformer(self.bert_model_name)
        return self._bert

    @property
    def nlp(self) -> Any:
        _check_spacy_imports()
        if self._nlp is None:
            self._nlp = spacy.load(self.spacy_model_name)
        return self._nlp

    def fit(self, x_data: Sequence[str], **kwargs: Any) -> BaseVectorizer[str]:
        return self
       

    def transform_preprocess(self, x_data: Sequence[str], **kwargs) -> np.ndarray:        
        x_data = [re.sub(r'(\. )+', ". ", doc_data) for doc_data in x_data]
        x_data = [re.sub(r'( ([\.\,\?\!]))+', r"\1", doc_data) for doc_data in x_data]
        x_data_sents = [[x.text for x in self.nlp(doc_data).sents] for doc_data in x_data]

        x_data_vector_sents = [self.bert.encode(doc_sentences) for doc_sentences in x_data_sents]
        x_data_vector_docs = [np.mean(sent_vectors, 0) for sent_vectors in x_data_vector_sents] # type: ignore
        return np.array(x_data_vector_docs)

    def transform(self, x_data: Sequence[str], **kwargs: Any) -> np.ndarray:
        x_data_vector_sents = self.bert.encode(x_data, show_progress_bar=True) # type: ignore
        return x_data_vector_sents # type: ignore
        
    def fit_transform(self, x_data: Sequence[str], **kwargs) -> np.ndarray:
        self.fit(x_data)
        return self.transform(x_data)

    def __getstate__(self):
        state = {key: value for (key, value) in self.__dict__.items() if key not in self.taboo_fields}
        state = {**state.copy(), **self.taboo_fields}
        return state