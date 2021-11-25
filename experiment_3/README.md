## Experiment 3

#### Dependencies
```sh
  instancelib
  python-allib
  scikit-learn
  matplotlib
  pandas
  numpy==1.20
```

#### Summary
Experiment 3, checking whether active learning reduces the amount of labeled data needed to get similar performance.

#### How to run

Go through *active_learning.ipynb* to see the effects of active learning.

Note that this is one run of the experiment, with one type of synthetic texts. In total, three runs were done: baseline (no synthetic texts), iVAE texts, and AugLy texts.

After identifying the minimum amount of labeled data needed, a machine learning baseline can be created using *ML_baseline.ipynb*. This shows whether active learning really needs less or that such a small amount of data would always have worked.
