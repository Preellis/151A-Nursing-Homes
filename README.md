# CSE 151A Nursing Homes

## Project Abstract
Nursing homes have a variety of factors that contribute to the number of COVID-19 cases making predicting case loads difficult. Using data science and machine learning methods, we aim to find strategies to predict nursing home COVID-19 cases. Using nursing home data from US Centers for Medicare and Medicaid Services, we want to predict the number of cases being treated inside of a nursing home during a given week based on various factors such as the number of total beds in the home, the number of residents who are up to date on COVID 19 vaccinations 14 days or more before their positive test, and the number of staff that week that have tested positive for COVID 19. To make these predictions, we will implement both linear regression and a neural network. Linear regression relies on a linear relationship between independent and dependent variables to predict unknown values, while neural networks may be a better choice for more complex datasets with a non-linear relationship between variables. We will compare the methods based on their respective accuracy, precision, and recall to better understand which machine learning model best fits the proposed problem. Our results will help us better understand the containment of the virus and may be useful in making healthcare decisions for the well-being of nursing home inhabitants.

## Data Exploration
[Notebook](https://github.com/Preellis/151A-Nursing-Homes/blob/main/main.ipynb)

We found the data column descriptions available [here](https://data.cms.gov/sites/default/files/2023-08/COVID-19%20Nursing%20Home%20Data%20Dictionary.pdf). For the non underfilled columns (which we dropped), we were able to get nearly 2 million completed data points of the roughly 3 million available. We decided against analyzing the non numerical data as there were too many categories to one hot encode and integer encoding did not make sense. Much of the features/target are either multimodal or weighted towards zero (particularly for our target of weekly resident cases per 1000 which is mostly 0s). We did realize some of our targets were over 1000 per 1000 but this is actually expected due to high turnover and the base '1000' being the number of occupied beds. Pairplots and correlation matrices are availble in the linked notebook for our data. 

## How We Will Preprocess Data
We will remove non-numerical columns (except for date which will be integer converted), drop majority empty columns, and drop all-time totals and columns already used in calculated columns. To deal with null/incomplete values, we will remove rows with incomplete or null data since we have plenty of data and over 50% of it is complete for the remaining columns. To make sure that the data works well when we apply it to networks, we will be MinMax scaling all the features and the expected outputs since the features/output are either multi-modal or are skewed making them non-normal.

<a target="_blank" href="https://colab.research.google.com/github/Preellis/151A-Nursing-Homes">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# Model 1: Linear Regression
Our training mean squared error for the linear regression model is 947.83, while our testing mean squared error is 951.18. This means that our model is underfitted since the MSE for training and testing are both very high. This underfitting is caused by our dataset having several dependent variables with complex relationships between them and our independent variable, and the linear regression model is not best suited for those types of complex relationships.

# Next Two Models
## Shallow Artificial Neural Network
An artificial neural network will be better suited for our research question because it can handle complex relationships between dependent variables and an independent variable. It has a hidden layer and uses inputs and weights, which allows it to learn patterns in data. Using non-linear activation functions will also help discover these patterns, which is something linear regression cannot do.
## Deep Neural Network
We will later transition to a deep neural network, which has more hidden layers than a shallow artificial neural network. Adding more layers will allow it to recognize more intricate patterns in the data. A DNN may overfit to the data if it is too complex for the data so it will be a good tool for comparison with a shallow ANN.
# Conclusion
Our linear regression model is not sufficient enough for our data because it has too many independent variables causing underfitting so the best solution to improve the performance would be to do feature expansion to explore non-linear options. In the next milestone, we will explore Neural Network models to find a better-fitting model. 
