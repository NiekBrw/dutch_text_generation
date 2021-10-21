#%%
import itertools
from os import PathLike
from allib.analysis.analysis import classifier_performance
from allib.environment.base import IT
from allib.environment.memory import MemoryEnvironment
from allib.activelearning.base import ActiveLearner
from allib.analysis.initialization import IdentityInitializer
from allib.analysis.plotter import AbstractPlotter
from allib.module.catalog import ModuleCatalog as Cat
from allib.stopcriterion.base import AbstractStopCriterion
from allib.module.factory import MainFactory
from allib.typehints.typevars import DT, RT
from instancelib.instances.base import InstanceProvider
from instancelib.labels.base import LabelProvider
from instancelib.typehints.typevars import KT, LT, VT
from instancelib.machinelearning.skdata import SkLearnDataClassifier
from instancelib.machinelearning import SkLearnVectorClassifier
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, Sequence, Tuple
import pandas as pd

from sklearn.pipeline import Pipeline # type: ignore
from sklearn.naive_bayes import MultinomialNB # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer # type: ignore

from instancelib.feature_extraction.textinstance import TextInstanceVectorizer
from instancelib.feature_extraction.textsklearn import SklearnVectorizer
from instancelib.functions.vectorize import vectorize
from instancelib.ingest.spreadsheet import read_csv_dataset, read_excel_dataset
from instancelib.instances.text import TextInstance
from instancelib.pertubations.base import TokenPertubator
from allib.configurations.ensemble import al_config_nb, tf_idf5000
from allib.analysis.simulation import initialize
from allib.stopcriterion.heuristic import AllDocsCriterion
from allib.analysis.simulation import multilabel_all_non_empty, simulate_with_cold_start
from allib.analysis.plotter import ClassificationPlotter

from sbert import PretrainedSentenceBERTVectorizer
#%%
tweet_env = read_csv_dataset("../active_learning_data_to_test_imbalanced.csv",
["clean_post"], ["set"])

#%%
# We create a train set of 70 %. 
# The remainder will be used as evaluation
train, test = tweet_env.train_test_split(tweet_env.dataset, 0.70)

#%%
# Model definitions
al_config_nb = {
    "paradigm": Cat.AL.Paradigm.LABEL_PROBABILITY_BASED,
    "query_type": Cat.AL.QueryType.LABELUNCERTAINTY,
    "label": "covid",
    "machinelearning": {
        "sklearn_model": Cat.ML.SklearnModel.SVC,
        "model_configuration": {
            "kernel": "linear", 
            "probability": True, 
            "class_weight": "balanced"
        },
        "task": Cat.ML.Task.BINARY,
        "balancer": {
            "type": Cat.BL.Type.IDENTITY,
            "config": {}
        }
    }
}

#%%
#%%
# 
sbert_vec = TextInstanceVectorizer(PretrainedSentenceBERTVectorizer("pdelobelle/robbert-v2-dutch-base"))
# The factory can build active learning objects according to model definitions
factory = MainFactory()
# Initializers can add prior knowledge to the model. 
# This initializer does not add any information
init = IdentityInitializer()
# The Instancelib Environment is slightly different.
# We add `labeled` and `unlabeled` sets by calling the following method
al_env = MemoryEnvironment.from_instancelib_simulation_heldout(
    tweet_env, train)
# This function actually builds the al and the feature extraction method
al, fe = initialize(factory, al_config_nb, 
    tf_idf5000, init, al_env)

#%%
# We vectorize the corpus via Robbert
vectorize(sbert_vec, al.env, fit = False)
#%%
# This object will track the classification performance
plotter = ClassificationPlotter(test, al.env.truth)

stop = AllDocsCriterion()

#%%
def simulate_classification(learner: ActiveLearner[IT, KT, DT, VT, RT, LT],
             stop_crit: AbstractStopCriterion[LT],
             plotter: AbstractPlotter[LT],
             batch_size: int, start_count=2) -> Tuple[ActiveLearner[IT, KT, DT, VT, RT, LT],
                         AbstractPlotter[LT]]:
    """Simulates the Active Learning procedure

    Parameters
    ----------
    learner : ActiveLearner[IT, KT, DT, VT, RT, LT]
        The Active Learning object
    stop_crit : AbstractStopCriterion[LT]
        The stopping criterion
    plotter : BinaryPlotter[LT]
        A plotter that tracks the results
    batch_size : int
        The batch size of each sample 
    start_count : int
        The number of instances that each class recieves before training the classification process. 

    Returns
    -------
    Tuple[ActiveLearner[IT, KT, DT, VT, RT, LT], AbstractPlotter[LT]]
        A tuple consisting of the final model and the plot of the process
    """
    learner.update_ordering()
    while not multilabel_all_non_empty(learner, start_count):
        instance = next(learner)
        oracle_labels = learner.env.truth.get_labels(instance)
        # Set the labels in the active learner
        learner.env.labels.set_labels(instance, *oracle_labels)
        learner.set_as_labeled(instance)
    while not stop_crit.stop_criterion:
        # Train the model
        learner.update_ordering()
        # Sample batch_size documents from the learner
        sample = itertools.islice(learner, batch_size)
        for instance in sample:
            # Retrieve the labels from the oracle
            oracle_labels = learner.env.truth.get_labels(instance)
            print(instance)
            print(oracle_labels)
            # Set the labels in the active learner
            learner.env.labels.set_labels(instance, *oracle_labels)
            learner.set_as_labeled(instance)

        plotter.update(learner)
        stop_crit.update(learner)
    
    return learner, plotter

#%%
simulate_classification(al, stop, plotter, 10, 2)
# %%
