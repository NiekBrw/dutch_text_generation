## iVAE<sub>MI</sub>

#### Dependencies
```sh
  tensorflow-gpu == 1.15
  torch==1.9.1
```

#### Summary
Create 21,000 synthetic texts for Experiments 2 and 3.

#### How to run

Preprocessing
```sh
  python preprocess_ivae_ex2.py --vocabfile data/vocab.dict --testfile data/test.txt --outputfile data
```

Generating
```sh
  python train_ivae_ex2.py --model mle_mi --test --train_from results_mle_mi/040.pt
```

#### Credits

Many thanks to the original [iVAE<sub>MI</sub>](https://github.com/fangleai/Implicit-LVM) GitHub repository, which is used for this part of the experiment.
