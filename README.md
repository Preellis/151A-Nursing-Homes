# CSE 151A Nursing Homes
[Notebook](https://github.com/Preellis/151A-Nursing-Homes/blob/main/main.ipynb)

## Project Abstract
Nursing homes have a variety of factors that contribute to the number of COVID-19 cases making predicting case loads difficult. Using data science and machine learning methods, we aim to find strategies to predict nursing home COVID-19 cases. Using nursing home data from US Centers for Medicare and Medicaid Services, we want to predict the number of cases being treated inside of a nursing home during a given week based on various factors such as the number of total beds in the home, the number of residents who are up to date on COVID 19 vaccinations 14 days or more before their positive test, and the number of staff that week that have tested positive for COVID 19. To make these predictions, we will implement both linear regression and 2 different neural networks: a shallow artificial neural network and a DNN. Linear regression relies on a linear relationship between independent and dependent variables to predict unknown values, while neural networks may be a better choice for more complex datasets with a non-linear relationship between variables. We will compare the methods based on their respective accuracy, precision, and recall to better understand which machine learning model best fits the proposed problem. Our results will help us better understand the containment of the virus and may be useful in making healthcare decisions for the well-being of nursing home inhabitants.

## Introduction
The breakout of the COVID-19 pandemic has disproportionately affected vulnerable populations, and nursing homes, housing elderly individuals with underlying health conditions, have been particularly susceptible to the spread of the virus. Studying past data and building precise predictive models will help in outbreak prevention, enabling better and faster responses. However, nursing homes have a variety of factors that contribute to the number of COVID-19 cases making predicting case loads difficult. Using data science and machine learning methods, we aim to find strategies to predict nursing home COVID-19 cases. Using nursing home data from the US Centers for Medicare and Medicaid Services, we want to predict the number of cases being treated inside of a nursing home during a given week based on various factors such as the number of total beds in the home, the number of residents who are up to date on COVID 19 vaccinations, the number of staff that week that have tested positive, and other factors. Getting a precise prediction is important because then departments can make more effective medical resource allocation according to the results. It can also help with the implementation of proactive and targeted measures for the prevention and containment of outbreaks during the pandemic. To make these predictions, we will implement both linear regression and 2 different neural networks: a simple artificial neural network and a DNN with dropout. Linear regression relies on a linear relationship between independent and dependent variables to predict unknown values, while neural networks may be a better choice for more complex datasets with a non-linear relationship between variables. We will compare the methods based on their respective accuracy, precision, and recall to better understand which machine learning model best fits the proposed problem. Our results will help us better understand the containment of the virus and may be useful in making healthcare decisions for the well-being of nursing home inhabitants.

## Methods

### Data Exploration

We found the data column descriptions available [here](https://data.cms.gov/sites/default/files/2023-08/COVID-19%20Nursing%20Home%20Data%20Dictionary.pdf). For the non underfilled columns (which we dropped), we were able to get nearly 2 million completed data points of the roughly 3 million available. We decided against analyzing the non numerical data as there were too many categories to one hot encode and integer encoding did not make sense. All of the features/target are either multimodal or weighted towards zero (particularly for our target of weekly resident cases per 1000 which is mostly 0s) or otherwise not normal according to the Shapiro scores. We did realize some of our targets were over 1000 per 1000 but this is actually expected due to high turnover and the base '1000' being the number of occupied beds. Pairplots and correlation matrices are availble in the linked notebook for our data. 

## Pre-Processing
We will remove non-numerical columns (except for date which will be integer converted), drop majority empty columns, and drop all-time totals and columns already used in calculated columns. To deal with null/incomplete values, we will remove rows with incomplete or null data since we have plenty of data and over 50% of it is complete for the remaining columns. To make sure that the data works well when we apply it to networks, we will be MinMax scaling all the features and the expected outputs since the features/output are either multi-modal or are skewed making them non-normal.

# Model 1: Linear Regression 




# Model 2: Simple Artificial Neural Network (Milestone 4):
Our data seemed to work for the new model (an ANN with 2 equal-sized hidden layers), and we decided to stick with MSE as our loss since it seems like the most useful for our data considering we are trying to predict the number of cases per week per 1000 residents. We did add a partition of the data for a validation set such that we could perform hyperparameter tuning. We saw 802.16 training MSE, 768.67 validation MSE, and 807.70 testing MSE with the model. The training, validation, and testing errors are fairly close suggesting they are not overfitting but they are still somewhat high suggesting it might be underfitting. Taking a look at the Training vs Validation error graph as seen in Fig 1, it seems to support that we are still underfitting since both training and validation errors continue to decrease (though fairly slowly) during the training epochs.  Both not training for enough epochs and the relatively low complexity of the model could be contributing to the model underfitting. Compared to the first model though, it is definitely less underfit (more towards the center of the fitting graph) as we see lower testing and training errors. In terms of optimization, we performed relatively course-grained hyperparameter tuning searching for the best units per hidden layer, activation function, and learning rate. We found that 14 units per hidden layer, sigmoid activation, and a learning rate of 0.01 resulted in the lowest validation loss.

![image](https://github.com/Preellis/151A-Nursing-Homes/assets/102556688/43c2562d-7f83-4163-8acb-ca55f351ed34)

 **Fig 1**: *Training and Validation Error over time for the Simple Artificial Neural Network*

 # Model #: Deep Artificial Neural Network


## Results

### Model 1: Linear Regression

Our training mean squared error for the linear regression model is 954.97, while our testing mean squared error is 951.46. This means that our model is underfitted since the MSE for training and testing are both very high. This underfitting is caused by our dataset having several dependent variables with complex relationships between them and our independent variable, and the linear regression model is not best suited for those types of complex relationships.


### Model 2: 


### Model 3





##Discussion

###idea
For this project,  the motivation is the critical need to develop proactive strategies in the face of the pandemic's disproportionate impact on vulnerable populations and the possibility of achieving more effective resource allocation.

###data exploration and preprocessing
As we gathered and preprocessed the data, we found out that a lot of attributes were corrupted, and some of them were not helpful for our desired prediction: the number of cases being treated inside a nursing home during a given week. As the corrupted data does not occupy a big percentage, we decided to drop corrupted and unrelated data. However, the data still demonstrated some points that are contrary to the expectation like the correlation between wklyResCasesPer1000 and percentStaffVaxUptoDate,  percentResVaxUpToDate and wklyResCasesPer1000 are 0.0. 

### model 1

### model 2

### model 3
Through each model, our goal was to increase the complexity to combat underfitting due to the nature of our features. With the increase in complexity, our model’s results became slightly more accurate with each step. Our mean squared error throughout each model never went below ____, revealing that it was difficult to take the minimal correlation found in pre-processing to an extremely accurate result. Nonetheless, the error improved upon complex models, reinstating the fact that our models were working to properly predict COVID cases with the power of the model’s respective complexity. 


## Conclusion
Our linear regression model is not sufficient enough for our data because it has too many independent variables causing underfitting so the best solution to improve the performance would be to do feature expansion to explore non-linear options. In the next milestone, we will explore Neural Network models to find a better-fitting model.

Our Artificial neural network may still not be sufficiently complex for the dataset and may not be able to learn the complex relationships between the features and our target. It still has a relatively high loss but did see improvement over the simpler linear regression model likely due to its additional complexity and ability to incorporate non-linearity thanks to the activation functions. The model was significantly slower than our first and more difficult to train which could be a concern if speed is an issue or if there is no access to a GPU. To improve, we could do more fine-tuned hyperparameter tuning and potentially add more epochs to the training to allow it to fit more to the dataset.



## Collaboration section
