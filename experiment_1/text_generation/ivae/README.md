## iVAE<sub>MI</sub>

#### Dependencies
```sh
  tensorflow-gpu == 1.15
  torch==1.9.1
```

#### How to run

Preprocessing
```sh
  python preprocess_ivae.py --trainfile data/train.txt --valfile data/val.txt --testfile data/test.txt --outputfile data
```

Training
```sh
  python train_ivae.py --model mle_mi
```

Testing
```sh
  python train_ivae.py --model mle_mi --test --train_from results_mle_mi/040.pt
```

#### Credits

Many thanks to the original [iVAE<sub>MI</sub>](https://github.com/fangleai/Implicit-LVM) GitHub repository, which is used for this part of the experiment.
