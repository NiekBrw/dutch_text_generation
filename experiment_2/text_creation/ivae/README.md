## iVAE<sub>MI</sub>

#### Dependencies
```sh
  tensorflow-gpu == 1.15
  torch==1.9.1
```

#### Summary
Train the iVAE<sub>MI</sub> method on Dutch tweet data.

Task for testing: generate 4000 samples based on the real test data.

#### How to run

Preprocessing
```sh
  python preprocess_ivae.py --vocabfile data/vocab.dict --testfile data/test.txt --outputfile data
```

Generating
```sh
  python train_ivae.py --model mle_mi --test --train_from results_mle_mi/040.pt
```

#### Credits

Many thanks to the original [iVAE<sub>MI</sub>](https://github.com/fangleai/Implicit-LVM) GitHub repository, which is used for this part of the experiment.
