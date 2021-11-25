from abc import abstractmethod
from typing import Any, Callable, Generic, Iterable, List
from instancelib.environment.base import AbstractEnvironment
from instancelib import InstanceProvider
from instancelib.pertubations.base import ProviderPertubator, ChildGenerator
from instancelib.typehints import KT, LT
from instancelib.feature_extraction.base import BaseVectorizer
from instancelib import TextInstance, Instance
from instancelib.utils.numpy import matrix_tuple_to_vectors
from instancelib.utils.to_key import to_key
import pandas as pd
import numpy as np
from random import sample
import random

np.random.seed(0)
random.seed(10)

class TextSynthesizer(ProviderPertubator[TextInstance[KT, np.ndarray], KT, str, np.ndarray, str], ChildGenerator[TextInstance[KT,np.ndarray]], Generic[KT, LT]):
    vectorizer: BaseVectorizer[Instance[KT, str, np.ndarray, str]]
    
    @abstractmethod
    def generate_data(self, ins: TextInstance[KT, np.ndarray]) -> Iterable[str]:
        """Generate synthetic data

        Parameters
        ----------
        ins : TextInstance[KT, np.ndarray]
            The input instance

        Returns
        -------
        Iterable[str]
            Synthesized data based on the input instance
        """        
        # Inherit and implement this function
        raise NotImplementedError

    def __call__(self, 
                 input: InstanceProvider[TextInstance[KT, np.ndarray], KT, str, np.ndarray, str]
                 ) -> InstanceProvider[TextInstance[KT, np.ndarray], KT, str, np.ndarray, str]:
        # Instances in the provider may already have been processed. 
        # We keep two lists where we accumulate these
        existing_keys: List[KT] = []
        new_generated: List[TextInstance[KT, np.ndarray]] = []
        for ins in input.values():
            # Check if we already have generated synthethic texts
            existing_generated = self.env.all_instances.get_children_keys(ins)
            if existing_generated:
                existing_keys = [*existing_keys, *existing_generated]
            else:
                generated_texts = self.generate_data(ins)
                # We give the new instance the same labels as the original
                original_labels = self.env.labels.get_labels(ins)
                for text in generated_texts:
                    # We create a new instance with the generated text
                    new_instance = self.env.create(text, None, text)
                    self.register_child(ins, new_instance)
                    new_generated.append(new_instance)
                    # We give the new instance the same labels as the original
                    self.env.labels.set_labels(new_instance, *original_labels)
        # We need to add vectors to the newly generated texts (if any)
        new_keys = [to_key(ins) for ins in new_generated]
        if new_generated:
            matrix = self.vectorizer.transform(new_generated)
            ret_keys, vectors = matrix_tuple_to_vectors(new_keys, matrix)
            self.env.add_vectors(ret_keys, vectors)
        # We return an InstanceProvider containing both the existing and newly generated 
        # synthetic examplse
        ins_keys =  existing_keys + new_keys
        new_provider = self.env.create_bucket(ins_keys)
        return new_provider

class PreSynthesized(TextSynthesizer[KT, LT]): 
    def __init__(self,
                 env: AbstractEnvironment[TextInstance[KT, np.ndarray], KT, str, np.ndarray, str, Any],
                 lookup_table: pd.DataFrame,
                 vectorizer: BaseVectorizer[Instance[KT, str, np.ndarray, str]],
                 n_synthetic_samples: int):
        self.env = env
        self.lookup_table = lookup_table
        self.vectorizer = vectorizer
        self.n_synthetic_samples = n_synthetic_samples

    def generate_data(self, ins: TextInstance[KT, np.ndarray]) -> Iterable[str]:
        subset = self.lookup_table[self.lookup_table.identifier == ins.identifier]
        texts: List[str] = subset.synthetic_text.to_list()
        if np.nan in texts:
            texts.remove(np.nan)
#         print(len(texts))
        try:
            texts = sample(texts, self.n_synthetic_samples)
        except: pass
        return texts