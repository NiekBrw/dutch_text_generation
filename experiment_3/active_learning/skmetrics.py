#%%

from typing import Any, FrozenSet, Sequence, Tuple
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.naive_bayes import MultinomialNB

import instancelib as il
from instancelib.machinelearning.sklearn import SkLearnClassifier
from instancelib.typehints.typevars import KT, LT
from instancelib.utils.func import list_unzip

#%% 
def sklearn_truth_pred(model: SkLearnClassifier[Any, KT, Any, Any, LT],
                       predictions: Sequence[Tuple[KT, FrozenSet[LT]]],
                       truth: il.LabelProvider[KT, LT]) -> Tuple[np.ndarray, np.ndarray]:
    keys, preds = list_unzip(predictions)
    truths = [truth.get_labels(key) for key in keys]
    y_pred = model.encoder.encode_batch(preds)
    y_true = model.encoder.encode_batch(truths)
    return y_true, y_pred

#%%
df = pd.read_csv("datasets/active_learning_data_to_test_imbalanced_new.csv")
tweet_env = il.pandas_to_env_with_id(df, "identifier", "clean_post", "set")
vect = il.TextInstanceVectorizer(il.SklearnVectorizer(TfidfVectorizer(max_features=1000)))
il.vectorize(vect, tweet_env)
train, test = tweet_env.train_test_split(tweet_env.dataset, 0.70)
model = il.SkLearnVectorClassifier.build(MultinomialNB(), tweet_env)
model.fit_provider(train, tweet_env.labels)
performance = il.classifier_performance(model, test, tweet_env.labels)
predictions = model.predict(test)
#%%
y_true, y_pred = sklearn_truth_pred(model, predictions, tweet_env.labels)
print(confusion_matrix(y_true, y_pred))
print(roc_auc_score(y_true, y_pred))