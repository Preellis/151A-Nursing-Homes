# CSE 151A Nursing Homes

**[Notebook](https://github.com/Preellis/151A-Nursing-Homes/blob/main/main.ipynb)**

## Introduction
The breakout of the COVID-19 pandemic has disproportionately affected vulnerable populations, and nursing homes, housing elderly individuals with underlying health conditions, have been particularly susceptible to the spread of the virus. Studying past data and building precise predictive models will help in outbreak prevention, enabling better and faster responses. However, nursing homes have a variety of factors that contribute to the number of COVID-19 cases making predicting case loads difficult. Using data science and machine learning methods, we aim to find strategies to predict nursing home COVID-19 cases. Using nursing home data from the US Centers for Medicare and Medicaid Services, we want to predict the number of cases being treated inside of a nursing home during a given week based on various factors such as the number of total beds in the home, the number of residents who are up to date on COVID 19 vaccinations, the number of staff that week that have tested positive, and other factors. Getting a precise prediction is important because then departments can make more effective medical resource allocation according to the results. It can also help with the implementation of proactive and targeted measures for the prevention and containment of outbreaks during the pandemic. To make these predictions, we will implement both linear regression and 2 different neural networks: a simple artificial neural network and a DNN with dropout. Linear regression relies on a linear relationship between independent and dependent variables to predict unknown values, while neural networks may be a better choice for more complex datasets with a non-linear relationship between variables. We will compare the methods based on their respective accuracy, precision, and recall to better understand which machine learning model best fits the proposed problem. Our results will help us better understand the containment of the virus and may be useful in making healthcare decisions for the well-being of nursing home inhabitants.

## Methods

### Data Exploration

We found the data column descriptions available [here](https://data.cms.gov/sites/default/files/2023-08/COVID-19%20Nursing%20Home%20Data%20Dictionary.pdf). For the non-underfilled columns (which we dropped), we were able to get nearly 2 million completed data points of the roughly 3 million available. We decided against analyzing the non-numerical data as there were too many categories to one hot encode and integer encoding did not make sense. All of the features/target are either multimodal or weighted towards zero (particularly for our target of weekly resident cases per 1000 which is mostly 0s) or otherwise not normal according to the Shapiro scores. We did realize some of our targets were over 1000 per 1000 but this is actually expected due to high turnover and the base '1000' being the number of occupied beds. Pairplots and correlation matrices are available in the linked notebook for our data. 

### Pre-processing
We removed non-numerical columns (except for date which was integer converted), dropped the majority empty columns, and dropped all-time totals and columns already used in calculated columns. We also decided to drop the columns that are not contributing to our desired prediction(the number of cases being treated inside of a nursing home during a given week), like the total amount in the dataset, or columns that are included in calculated percentages. To deal with null/incomplete values, we will remove rows with incomplete or null data since we have plenty of data and over 50% of it is complete for the remaining columns. To make sure that the data works well when we apply it to networks, we will be MinMax scaling all the features and the expected outputs since the features/output are either multi-modal or are skewed making them non-normal. For train-test split, we had a ratio of 60:20:20 for training, testing, and validation respectively, and a random state of 21.

### Model 1: Linear Regression
We chose our first model to be linear regression because we wanted to examine if there was a linear relationship between various factors of a nursing home and how many weekly COVID cases for every thousand occupied beds.
```
reg = LinearRegression()
regmodel = reg.fit(X_train, y_train)
```


### Model 2: Simple Artificial Neural Network (Milestone 4):

For our next model, we used a Simple (or Shallow) Artificial Neural Network composed of two hidden dense layers. We performed hyperparameter tuning to search for the best units per hidden layer, activation function and learning rate. This resulted in a neural network with 14 units per hidden layer, a sigmoid activation function and a learning rate of 0.01, as these parameters resulted in the smallest validation loss. This model can most easily be visualized like model __. This model was trained for 20 epochs with a batch size of 100 on the training data, and then tested on the validation and test datasets. 

[image]

 ### Model #3: Deep Artificial Neural Network

For our final model, we used a Deep Artificial Neural Network composed of four hidden dense layers and an additional dropout layer. Similar to the Simple Neural Network, we performed hyperparameter turning to search for the best units per dense hidden layer, activation function and learning rate. This resulted in a neural network with 28 units per dense hidden layer, a ‘relu’ activation function and a learning rate of 0.001. After these dense layers, a dropout layer with a rate of 0.2 was added to prevent overfitting. This model can most easily be visualized like model __. This model was trained for 40 epochs and with a batch size of 100 on the training data, and then tested on the validation and test datasets.


![Model Performance](https://cdn.discordapp.com/attachments/1217549861935775885/1217627186240950302/image.png?ex=6604b6af&is=65f241af&hm=2ea73d679df107fd75d0013eba20ea87e9af5ec54b1107b68c734c269c85f5f0&)

## Results

### Model 1: Linear Regression

Our training mean squared error for the linear regression model is 954.97, while our testing mean squared error is 951.46. 

### Model 2: 

 We saw the following training and validation error throughout the epochs as seen in Fig.1. This resulted in a Training MSE of 802.16 , a validation MSE of 768.67,  and a testing MSE of 807.70. l

![image](https://github.com/Preellis/151A-Nursing-Homes/assets/102556688/43c2562d-7f83-4163-8acb-ca55f351ed34)

 **Fig 1**: *Training and Validation Error over time for the Simple Artificial Neural Network*

### Model 3

We saw the following training and validation error throughout the epochs as seen in Fig. 2. This resulted in a Training MSE of 793.34, a validation MSE of 763.74 and testing MSE of 801.45. 
 
[image]
**Fig 2**: *Training and Validation Error over time for the Deep Artificial Neural Network*

## Discussion

### Idea
For this project,  the motivation is the critical need to develop proactive strategies in the face of the pandemic's disproportionate impact on vulnerable populations and the possibility of achieving more effective resource allocation.

### data exploration and preprocessing
As we gathered and preprocessed the data, we found out that a lot of attributes were corrupted, and some of them were not helpful for our desired prediction: the number of cases being treated inside a nursing home during a given week. As the corrupted data does not occupy a big percentage, we decided to drop corrupted and unrelated data. However, the data still demonstrated some points that are contrary to the expectation like the correlation between wklyResCasesPer1000 and percentStaffVaxUptoDate,  percentResVaxUpToDate and wklyResCasesPer1000 are 0.0. 

### Model 1: Linear Regression

Looking at the results of the linear regression,  our model is underfitted since the MSE for training and testing are both very high. This underfitting is caused by our dataset having several dependent variables with complex relationships between them and our independent variable, and the linear regression model is not best suited for those types of complex relationships.


### Model 2: Simple Artificial Neural Network
For the second model we were building an Simple Artificial Neural Network and decided to use tuning to find out the best hypermeter set. However, we decided to limit the amount of combinations to a relatively small number by limiting the number of choice of the nodes, learning rate, and activation function, as too many choices resulting in a really long runtime. After research we agree on the number of nodes should at least be (input node + output node)/2 = (13+1)/2 = 7 nodes, thus we gave our tuner choice of 7, 14, 28. We also select choice of learning rate of 0.1, 0.01, 0.001 and common activation functions, making an combination fo 3*3*3 = 27.(our maximum trial). After running the turner we get the best perfoming set of hyperparameter among the 27 sets, and the result is quite believable: the performance got better than model1. 

### Model 3: Deep Artificial Neural Network




## Conclusion
Our linear regression model is not sufficient enough for our data because it has too many independent variables causing underfitting so the best solution to improve the performance would be to do feature expansion to explore non-linear options. In the next milestone, we will explore Neural Network models to find a better-fitting model.

Our Artificial neural network may still not be sufficiently complex for the dataset and may not be able to learn the complex relationships between the features and our target. It still has a relatively high loss but did see improvement over the simpler linear regression model likely due to its additional complexity and ability to incorporate non-linearity thanks to the activation functions. The model was significantly slower than our first and more difficult to train which could be a concern if speed is an issue or if there is no access to a GPU. To improve, we could do more fine-tuned hyperparameter tuning and potentially add more epochs to the training to allow it to fit more to the dataset.





Through each model, our goal was to increase the complexity to combat underfitting due to the nature of our features. With the increase in complexity, our model’s results became slightly more accurate with each step. Our mean squared error throughout each model never went below 750, revealing that it was difficult to take the minimal correlation found in pre-processing to an extremely accurate result. Nonetheless, the error improved upon complex models, reinstating the fact that our models were working to properly predict COVID cases with the power of the model’s respective complexity. 

## Collaboration section
