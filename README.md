# Credit Scoring Models

## Introduction

Credit scoring models are essential tools in the financial industry used to assess the creditworthiness of individuals and organizations. By analyzing historical data and identifying patterns, these models predict the likelihood of a borrower defaulting on their loan obligations. This enables lenders to make informed decisions on loan approvals, interest rates, and credit limits.

## Task

The objective is to use the loan data in order to train models that can predict whether a loan will be fully paid off or charged off. The secondary objective is to determine the factors which affect the predictions of the model. For this purpose we will leverage a Logistic Regression, a Random Forest and XGBoost, while also trying to understand the importance of each feature of the dataset.

## Instructions

The first step is to create an environment using the .yml provided as follows:

`conda env create -f env.yml`

Once the environment is created, you can run the main.py script according to your preferences by changing the input arguments:

`python main.py`

## Results

The following results show that XGBoost outperforms the other other models. To measure the performance, we use the Gini coefficient, Kolomogorov-Smirnov statistic and AUC metrics.

![alt text](https://github.com/apostolikas/Credit-Scoring-Models/blob/main/images/Gini_KS.png)
![alt text](https://github.com/apostolikas/Credit-Scoring-Models/blob/main/images/aucs.png)

By accessing the feature importance for each model, we can conclude that the attribute 'age' and 'subgrade' are the most important ones.

While performance is what we care about the most, we should also opt for developing fair ML solutions. To this end, we computed the Demographic Parity Difference (DPD) metric to have an idea of how fair each model is. We select 'age' to be the sensitive attribute.

| Model | DPD   |
| :---:   | :---: |
| Logistic Regression | 0.0826   |
| Random Forest | 0.0885   |
| XGBoost | 0.1102   |

On average the privileged group (age above the median) has a 9.37% higher rate of receiving the positive outcome than the unprivileged group.
