## General Data Processes

### Data Collection
#### Dependencies
```sh
  snscrape
  pandas
```  

#### Summary
Collecting, pseudonimizing and saving Dutch tweets on COVID-19 and the benefits affair (_"Toeslagenaffaire"_).

#### How to run?
Go through the data_collection.ipynb notebook.

### Data Inspection
#### Dependencies
```sh
  pandas
  numpy
  ipywidgets
  pandas_profiling
  seaborn
  matplotlib
  plotly
  nltk
  wordcloud
  palettable
```  

#### Summary
Inspecting the tweet data. Looking at statistics, users, timelines and wordclouds.

#### How to run?
Go through the data_inspection.ipynb notebook.

### Data Preprocessing
#### Dependencies
```sh
  pandas
  emoji
```  

#### Summary
General preprocessing of the data, by removing duplicates and overlap in the majority and minority sets. Also removing emojis and some of the user mentions and hashtags, and only keeping tweets that are between 2 and 40 words after preprocessing.

Experiment 1 preprocessing by replacing tokens with ids for ARAML.

Experiment 2 preprocessing ?

Creating all datasets for the experiments.

#### How to run?
Go through the data_preprocessing.ipynb notebook.
