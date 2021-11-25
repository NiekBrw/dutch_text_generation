## Creating Synthetic Text

#### Dependencies
```sh
  augly
  pandas
  nltk
```

#### Summary
Create synthetic texts for Experiments 2 and 3.

For both iVAE and AugLy, 21,000 synthetic texts have to be created.

### How to run

To generate the iVAE synthetic texts, go to *ivae/*.

With these raw texts, *ex2_text_generation.ipynb* can be used to clean them, generate AugLy texts and prepare the datasets that will be used in the experiment.

*ex2_text_evaluation.ipynb* evaluates the quality and diversity of the synthetic texts.
