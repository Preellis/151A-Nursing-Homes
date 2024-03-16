# 🔮A Step Ahead🔮: Predicting COVID-19’s Spread in Nursing Homes

**[Notebook](https://github.com/Preellis/151A-Nursing-Homes/blob/main/main.ipynb)**

## Introduction
The breakout of the COVID-19 pandemic has disproportionately affected vulnerable populations. Nursing homes, housing elderly individuals often with underlying health conditions, have been particularly susceptible to the spread of the virus. Analyzing past data and building precise predictive models will help in outbreak prevention, enabling better and faster responses. However, nursing homes have a variety of factors that contribute to the number of COVID-19 cases, making predicting case loads difficult. Using data science and machine learning methods, we aim to find strategies to predict nursing home COVID-19 cases. Using nursing home data from the US Centers for Medicare and Medicaid Services, we want to predict the number of COVID-19 cases inside a nursing home during a given week based on various factors such as the number of total beds in the home, the number of residents who are up to date on COVID 19 vaccinations, and the number of staff that week that have tested positive. Getting a precise prediction is important because it will allow departments to make more effective medical resource allocations according to the results. It can also help with the implementation of proactive and targeted measures for the prevention and containment of outbreaks during the pandemic. To make these predictions, we will implement both linear regression and two different neural networks: a simple artificial neural network and a deep neural network with dropout. Linear regression relies on a linear relationship between independent and dependent variables to predict unknown values, while neural networks may be a better choice for more complex datasets with a non-linear relationship between variables. We will compare the methods based on their training, validation, and test mean squared error to better understand which machine learning model best fits the proposed problem. Our results will help us better understand how to contain the virus and will be useful in making healthcare decisions for the well-being of nursing home inhabitants.

## Methods

### Data Exploration

We found the data column descriptions available [here](https://data.cms.gov/sites/default/files/2023-08/COVID-19%20Nursing%20Home%20Data%20Dictionary.pdf). For the non-underfilled columns (which we dropped), we were able to get nearly 2 million completed data points of the roughly 3 million available. We decided against analyzing the non-numerical data as there were too many categories to one hot encode and integer encoding did not make sense. All of the features/target are either multimodal or weighted towards zero (particularly for our target of weekly resident cases per 1000 which is mostly 0s) or otherwise not normal according to the Shapiro scores. We did realize some of our targets were over 1000 per 1000 but this is actually expected due to high turnover and the base '1000' being the number of occupied beds. Pairplots and correlation matrices are available in the linked notebook for our data.

### Pre-processing
We removed non-numerical columns (except for date which was integer converted), dropped the empty columns, and dropped all-time totals and columns already used in calculated columns. We also decided to drop the columns that are not contributing to our desired prediction(the number of cases being treated inside of a nursing home during a given week), like the total amount in the dataset, or columns that are included in calculated percentages. To deal with null/incomplete values, we will remove rows with incomplete or null data since we have plenty of data and over 50% of it is complete for the remaining columns. To make sure that the data works well when we apply it to networks, we will be MinMax scaling all the features and the expected outputs since the features/output are either multi-modal or are skewed making them non-normal. For train-test split, we had a ratio of 60:20:20 for training, testing, and validation respectively, and a random state of 21.

```
|         |      date |   wklyResDeaths |   numBeds |   numBedsUsed |   resHospitalCases |   resVaxxedHospitalCases |   wklyStaffCases |   wklyResCovidDeathsPer1000 |   wklyRes |   wklyStaff |   percentStaffVaxxedBefore |   percentResVaxUpToDate |   percentStaffVaxUptoDate |
|--------:|----------:|----------------:|----------:|--------------:|-------------------:|-------------------------:|-----------------:|----------------------------:|----------:|------------:|---------------------------:|------------------------:|--------------------------:|
|  585975 | 0.468431  |               0 | 0.0566686 |     0.0215774 |                  0 |                        0 |            0.004 |                           0 | 0.0232009 |   0.0529671 |                    0.90741 |                 0       |                   0       |
| 1847360 | 0.0282112 |               0 | 0.0560962 |     0.03125   |                  0 |                        0 |            0     |                           0 | 0.0334251 |   0.0627759 |                    0.94531 |                 0       |                   0       |
|  425098 | 0.506884  |               0 | 0.079565  |     0.0345982 |                  0 |                        0 |            0     |                           0 | 0.0377507 |   0.0362923 |                    0.86486 |                 0.73958 |                   0.86486 |
| 1476017 | 0.98517   |               0 | 0.0417859 |     0.0256696 |                  0 |                        0 |            0.004 |                           0 | 0.0271333 |   0.0514958 |                    0.49524 |                 0.91304 |                   0.49524 |
|  860192 | 0.496788  |               0 | 0.0692616 |     0.0364583 |                  0 |                        0 |            0     |                           0 | 0.0389304 |   0.0711133 |                    0.95862 |                 0.76768 |                   0.50345 |
```

 **Table 1**: *A random sample of our cleaned dataframe, focusing on the features that we intend to use for our models. Columns are (from left to right): id of row, date, weekly resident deaths, number of beds, number of beds used, residents in the Hospital, residents that are in the hospital and up to date on their vaccines, weekly number of COVID cases among the staff, weekly COVID deaths among residents per 1000  occupied beds, number of residents staying at least one day in the nursing home in that week, number of staff working in the nursing home at least one day that week, percentage of staff that have received at least one COVID vaccination, percentage of residents who are up to date with vaccinations and percentage of staff who are up to date with their vaccinations.*



```
|         |   wklyResCasesPer1000 |
|--------:|----------------------:|
|  585975 |                 67.8  |
| 1847360 |                  0    |
|  425098 |                  0    |
| 1476017 |                 28.57 |
|  860192 |                 20.2  |
```
 **Table 2**: *A random sample of the ‘output’ of our cleaned dataframe, representing the weekly COVID-19 cases per thousand occupied beds for the five random columns. This matches up with Table 1.*
### Model 1: Linear Regression
We chose our first model to be linear regression because we wanted to examine if there was a linear relationship between various factors of a nursing home and how many weekly COVID cases for every thousand occupied beds.
```
reg = LinearRegression()
regmodel = reg.fit(X_train, y_train)
```


### Model 2: Simple Artificial Neural Network (Milestone 4):

For our next model, we used a Simple (or Shallow) Artificial Neural Network composed of two hidden dense layers. We performed hyperparameter tuning to search for the best units per hidden layer, activation function, and learning rate. This resulted in a neural network with 14 units per hidden layer, a sigmoid activation function, and a learning rate of 0.01, as these parameters resulted in the smallest validation loss. This model can most easily be visualized like model __. This model was trained for 20 epochs with a batch size of 100 on the training data, and then tested on the validation and test datasets.

![image](https://media.discordapp.net/attachments/1217549861935775885/1217635454044868668/image.png?ex=6604be62&is=65f24962&hm=bdf07425d2e10cacb76f8f518ff58641bf907cc8cd44a6772fa0fb87e802483c&=&format=webp&quality=lossless&width=1345&height=894)

 **Fig 1**: *Simple Artificial Neural Network Diagram, describing the structure and make up of our our second model*

 ### Model #3: Deep Artificial Neural Network

For our final model, we used a Deep Artificial Neural Network composed of four hidden dense layers and an additional dropout layer. Similar to the Simple Neural Network, we performed hyperparameter turning to search for the best units per dense hidden layer, activation function, and learning rate. This resulted in a neural network with 28 units per dense hidden layer, a ‘ReLU’ activation function, and a learning rate of 0.001. After these dense layers, a dropout layer with a rate of 0.2 was added to prevent overfitting. This model can most easily be visualized like model __. This model was trained for 40 epochs with a batch size of 100 on the training data, and then tested on the validation and test datasets.

![image](https://media.discordapp.net/attachments/1217549861935775885/1217638206573641848/image.png?ex=6604c0f3&is=65f24bf3&hm=eeef8e914558a05822703c2f5c194a95732a61a3b728d048ce17e00b5cab5449&=&format=webp&quality=lossless&width=1872&height=588)

 **Fig 2**: *Deep Artificial Neural Network Diagram, describing the structure and make up of our third model. *

## Results

### Model 1: Linear Regression

Our training mean squared error for the linear regression model is 954.97, while our validation mean squared error is 926.85, and out test mean squared error is 951.46

### Model 2: 

 We saw the following training and validation error throughout the epochs as seen in Fig.3. This resulted in a Training MSE of 802.51, a validation MSE of 767.20,  and a testing MSE of 807.91.

![image](https://cdn.discordapp.com/attachments/1197674736608628779/1217642064607580240/image.png?ex=6604c48a&is=65f24f8a&hm=be24418faa62dc8106b08be11081ca7ab647d9861128c704167c28a529c5b1c1&)

 **Fig 3**: *The Training and Validation Error over time (epochs) for our Simple Artificial Neural Network*

### Model 3

We saw the following training and validation error throughout the epochs as seen in Fig. 4. This resulted in a Training MSE of 793.34, a validation MSE of 763.74 and testing MSE of 801.45. 
 
![image](https://cdn.discordapp.com/attachments/1197674736608628779/1217642203820724224/image.png?ex=6604c4ac&is=65f24fac&hm=88b1acae1fe002eeafd1cf268daea10381d9cb01421878a4c450ae4c1b83ec92&)
**Fig 4**: *The Training and Validation Error over time (epochs) for our Deep Artificial Neural Network*


### Final Results Summary

![Model Performance](https://cdn.discordapp.com/attachments/1217549861935775885/1217627186240950302/image.png?ex=6604b6af&is=65f241af&hm=2ea73d679df107fd75d0013eba20ea87e9af5ec54b1107b68c734c269c85f5f0&)


## Discussion

### Idea
For this project, the motivation is the critical need to develop proactive strategies in the face of the pandemic's disproportionate impact on vulnerable populations and the possibility of achieving more effective resource allocation.

### Data Exploration and Preprocessing
As we gathered and preprocessed the data, we found out that a lot of attributes were corrupted, and some of them were not helpful for our desired prediction: the number of cases being treated inside a nursing home during a given week. As the corrupted data does not occupy a big percentage, we decided to drop corrupted and unrelated data. However, the data still demonstrated some points that are contrary to the expectation like the correlation between wklyResCasesPer1000 and percentStaffVaxUptoDate, percentResVaxUpToDate and wklyResCasesPer1000 are 0.0.

### Model 1: Linear Regression

We chose linear regression as our first model due to its simplicity and its ability to determine if there is a linear relationship between dependent variables and an independent variable. Looking at the results of the linear regression,  our model is underfitted since the MSE for training and testing are both very high. This underfitting is caused by our dataset having several dependent variables with complex relationships between them and our independent variable, and the linear regression model is not best suited for those types of complex relationships.


### Model 2: Simple Artificial Neural Network
For the second model we built a Simple Artificial Neural Network. This model will better show us the complex non-linear relationships that were not properly considered in the linear model. Using dense hidden layers will combat the underfitting that was seen in our linear regression model. 
 	When creating the model, we wanted to better grasp the and decided to use tuning to find out the best hypermeter set. However, we decided to limit the amount of combinations to a relatively small number by limiting the number of choices of the nodes, learning rate, and activation function, as too many choices resulted in a really long runtime. After research we agreed on the number of nodes should at least be (input node + output node)/2 = (13+1)/2 = 7 nodes, thus we gave our tuner choice of 7, 14, 28. We also select choice of learning rate of 0.1, 0.01, 0.001 and common activation functions, making an combination of 3*3*3 = 27.(our maximum trial). After running the tuner we get the best-performing set of hyperparameters among the 27 sets, and the result is quite believable: the performance got better than model1.

### Model 3: Deep Artificial Neural Network
For our third model, we built a Deep Artificial Neural Network. Analyzing our simple neural network showed us that our model was still underfitted, meaning that we needed to increase the complexity of our model. This was our reason for doubling the hidden dense layers and epochs to allow the network to better model the complex relationships. We decided to keep similar hyperparameter options as our second model, resulting in a slightly better performance than before.

## Conclusion
Our linear regression model is not sufficient enough for our data because it has too many independent variables causing underfitting so the best solution to improve the performance would be to do feature expansion to explore non-linear options. In the next milestone, we will explore Neural Network models to find a better-fitting model.

Our Artificial neural network may still not be sufficiently complex for the dataset and may not be able to learn the complex relationships between the features and our target. It still has a relatively high loss but did see improvement over the simpler linear regression model likely due to its additional complexity and ability to incorporate non-linearity thanks to the activation functions. The model was significantly slower than our first and more difficult to train which could be a concern if speed is an issue or if there is no access to a GPU. To improve, we could do more fine-tuned hyperparameter tuning and potentially add more epochs to the training to allow it to fit more to the dataset.





Through each model, our goal was to increase the complexity to combat underfitting due to the nature of our features. With the increase in complexity, our model’s results became slightly more accurate with each step. Our mean squared error throughout each model never went below 750, revealing that it was difficult to take the minimal correlation found in pre-processing to an extremely accurate result. Nonetheless, the error improved upon complex models, reinstating the fact that our models were working to properly predict COVID cases with the power of the model’s respective complexity. 

## Collaboration section
We worked as a team with a fairly flat team structure!

Preston Ellis -- Project Coder
Trained models locally
Organized team Discord
Coded for all milestones
Wrote portions of and edited milestone READMEs
Designed data preprocessing and model training
Organized team Github

Jackie Piepkorn -- Team Organizer
Wrote the abstract
Coded Milestone 2 and wrote a significant portion of the summary for the README
Organized group meetings
Reached out to group members via email and added everyone to Discord
Wrote part of methods and made graphs for writeup

Steve Yin -- Project Coder
Trained model locally
Wrote portions of the final report
Coded for milestones

Dante Testini -- Project Analyst 
Wrote significant portion of methods, including making diagrams, for the README
Wrote significant part of discussion section of README
Organized group meetings
Active in group discussions

Ryan Martinez -- None
Attended one meeting and wrote the Milestone 3 conclusion.

Sindhu Kothe -- None
Attended one meeting
