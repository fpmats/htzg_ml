# HTZG_ML
This repo contains all the machine-learning components of "A statistical analysis of phonon-induced bandgap renormalization within a database of 2D materials" (upcoming). 

HTZG is an acronym for **H**igh-**T**hroughput **Z**acharias-**G**iustino. High-throughput refers to the nature of the study, Zacharias-Giustino refers to the old name for the Special-Displacement Method and its primary authors. 

# Requirements 
* Code was tested on Python 3.10.9. 
* Users can directly install the correct version of the Python packages through the following commands:
```
pip install -r requirements.txt
```
# Linear Regression

We compare our more sophisticated machine learning model to ordinary least-squares linear regression. To generate plots and weights, run 

```
python htzg_linear_regression.py
```

# Extremely Randomized Trees (ERT)

Our main machine learning model are Extremely Randomized Trees (ERT)s as seen in [1]. The hyperparameter optimization step and model training are split into one script, which saves the final full-size and reduced models, and showing model performance on our data into another. 

To train the model from scratch, including hyperparameter optimization, run 

```
python htzg_train_ert.py
```

# References 

1: Geurts, Pierre, Damien Ernst, and Louis Wehenkel. "Extremely randomized trees." Machine learning 63 (2006): 3-42.