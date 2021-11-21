import instancelib as il
from instancelib.utils.func import union
from instancelib.typehints import KT, DT, VT, RT, LT
import random

random.seed(10)

from typing import Any, Tuple, TypeVar, Union
IT = TypeVar("IT", bound="il.Instance[Any, Any, Any, Any]", covariant=True)

def stratified_train_test(env: il.Environment[IT, KT, DT, VT, RT, LT], 
                          source: il.InstanceProvider[IT, KT, DT, VT, RT],
                          labels: il.LabelProvider[KT, LT],
                          train_size: Union[float, int]) -> Tuple[
                            il.InstanceProvider[IT, KT, DT, VT, RT],
                            il.InstanceProvider[IT, KT, DT, VT, RT]
                          ]:
    """Create a stratified train test split that respects the class ratios

    Parameters
    ----------
    env : il.Environment[IT, KT, DT, VT, RT, LT]
        The environment
    source : il.InstanceProvider[IT, KT, DT, VT, RT]
        The source provider
    labels : il.LabelProvider[KT, LT]
        The labelprovider
    train_size : Union[int, float]
        The ratio or absolute size of the train set

    Returns
    -------
    Tuple[ il.InstanceProvider[IT, KT, DT, VT, RT], il.InstanceProvider[IT, KT, DT, VT, RT] ]
        A tuple consisting of the train and test InstanceProviders
    """    
    labelset = labels.labelset
    n = len(source)
    ratio = 0.00
    if isinstance(train_size, int):
        ratio = train_size / len(source)
    else:
        ratio = train_size
    target_test_size = int((1 - ratio) * n)
    lbl_inss_map = {
        label: labels.get_instances_by_label(label).intersection(source) for label in labelset}
    ratios = {label: len(inss) / n for label, inss in lbl_inss_map.items()}
    goals = {label: int(ratio * target_test_size) for label, ratio in ratios.items()}
    samples = {label: frozenset(random.sample(inss, goals[label])) for label, inss in lbl_inss_map.items()}
    test_set = union(*samples.values())
    train_set = frozenset(source).difference(test_set)
    train_provider = env.create_bucket(train_set)
    test_provider = env.create_bucket(test_set)
    return train_provider, test_provider
    

