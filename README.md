# Anticipating Burnout using Work, Screen, and Social Habits
Group project for Team 9, CSCI 635, Intro to Machine Learning, Spring 2026.
- Vrutant Chaudhari
- Will Hoover
- Rina Peshori

## Abstract
  
In this project, we develop and fine-tune three models that classify a person's risk of burnout as low, medium, or high based on several factors. By correctly and applying principles of preprocessing and parameter fine-tuning, our models all achieved at least <average metric> on the test dataset, with our best model averaging <best metric>. Such models allow us to anticipate burnout before it happens, identify the most influential factors that may cause burnout, and recommend preemptive actions to reduce burnout risk.

## Execution instructions

Before running any code, ensure you have python downloaded on your machine. In the root directory of the project, start by installing all of the dependencies in the requirements:
`pip install -r requirements.txt`

To run the 3-model pipeline, simply run `python main.py` in the root directory. This will run all 3 models on a train-test split and output the accuracy and confusion matrix results.

Alternatively, `python main.py --kfold` will run k-fold cross-validation (number of folds is set to 10) on each model and output the training and validation accuracies.
