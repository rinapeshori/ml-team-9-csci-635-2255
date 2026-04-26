# Anticipating Burnout using Work, Screen, and Social Habits
Group project for Team 9, CSCI 635, Intro to Machine Learning, Spring 2026.
- Vrutant Chaudhari
- Will Hoover
- Rina Peshori

## Abstract
  
In this project, we develop and fine-tune three models that classify a person's risk of burnout as low, medium, or high based on several factors. By correctly and applying principles of preprocessing and parameter fine-tuning, our models all achieved at least 88% accuracy on the unaugmented test dataset, with our best model averaging 96.9% accuracy. Additionally, the three models averaged approximately 90% accuracy on the augmented dataset. High generalizability was noted, as no significant change in metrics was observed when using k-fold validation. Such models allow us to anticipate burnout before it happens, identify the most influential factors that may cause burnout, and recommend preemptive actions to reduce burnout risk.

## Execution instructions

Before running any code, ensure you have python downloaded on your machine. In the root directory of the project, start by installing all of the dependencies in the requirements:
`pip install -r requirements.txt`

To run the pipeline, run `python code/main.py` with one of the model flags in the root directory. The code will not run if no flag is specified. The model flags are as follows:
- `--mlp`: multi-level perceptron
- `--logistic`: logistic regression
- `--random_forest`: random forest
- `--all`: run all models
This will create a train/test split, train the specified model(s) and output the resulting confusion matrices and metrics.

Additional flags can be added to use other training methodologies:
- `--k-fold`: instead of just a train/test split, validate the chosen model(s) with 10-fold cross-validation. This will take a while for the random forest.
- `--augment`: Use the stratified augmented data instead of the original dataset.
