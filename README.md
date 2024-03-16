# 🔮A Step Ahead🔮: Predicting COVID-19’s Spread in Nursing Homes

**[Notebook](https://github.com/Preellis/151A-Nursing-Homes/blob/main/main.ipynb)**

## Introduction
The breakout of the COVID-19 pandemic has disproportionately affected vulnerable populations. Nursing homes, housing elderly individuals often with underlying health conditions, have been particularly susceptible to the spread of the virus. Analyzing past data and building precise predictive models will help in outbreak prevention, enabling better and faster responses. However, nursing homes have a variety of factors that contribute to the number of COVID-19 cases, making predicting case loads difficult. Using data science and machine learning methods, we aim to find strategies to predict nursing home COVID-19 cases. Using nursing home data from the US Centers for Medicare and Medicaid Services, we want to predict the number of COVID-19 cases inside a nursing home during a given week based on various factors such as the number of total beds in the home, the number of residents who are up to date on COVID 19 vaccinations, and the number of staff that week that have tested positive. Getting a precise prediction is important because it will allow departments to make more effective medical resource allocations according to the results. It can also help with the implementation of proactive and targeted measures for the prevention and containment of outbreaks during the pandemic. To make these predictions, we will implement both linear regression and two different neural networks: a simple artificial neural network and a deep neural network with dropout. Linear regression relies on a linear relationship between independent and dependent variables to predict unknown values, while neural networks may be a better choice for more complex datasets with a non-linear relationship between variables. We will compare the methods based on their training, validation, and test mean squared error to better understand which machine learning model best fits the proposed problem. Our results will help us better understand how to contain the virus and will be useful in making healthcare decisions for the well-being of nursing home inhabitants.

## Methods

### Data Exploration

We found the data column descriptions available [here](https://data.cms.gov/sites/default/files/2023-08/COVID-19%20Nursing%20Home%20Data%20Dictionary.pdf) which were fairly intuitive except for the denominators of the per 1000 calculated columns being based upon the number of occupied beds. For the non-underfilled columns (which we dropped), we were able to find nearly 1,987,815 completed data points of the roughly 2,923,332 available. The non-numerical data (which we did not use besides date) had 50+ distinct values per category. We checked the Shapiro scores of all the possibly useful columns and found all to have scores less than .05. When looking for outliers using histograms and the pandas describe function we did realize some of our targets were over 1000 per 1000 occupied beds as patient rotation in and out weekly skewed results. We also constructed a pair plot and correlation heatmap for our planned to be used features. Higher resolution graphs and descriptive statistics are available in the “Data Exploration/Visualization” section of our notebook linked at the top of this report


![image](https://cdn.discordapp.com/attachments/1217549861935775885/1218396110989819994/image.png?ex=660782cd&is=65f50dcd&hm=64ab1aac1f53a66fc275ce7b0088ba6190dae46ad78b605b29263099df0f084b&)

 **Fig 1**: *Histogram with logarithmic y scale (omitting 0s) for the weekly resident cases of COVID-19 per 1000 occupied beds in nursing homes according to the Centers for Medicare & Medicaid Services records*


![image](https://cdn.discordapp.com/attachments/1217549861935775885/1218395997080653866/image.png?ex=660782b2&is=65f50db2&hm=d45a9a7479e01fd989c0504f5ed17942d29e2d66e9b3a2ffc81c2f671edb0fd0&)

 **Fig 2**: *Pair plot of features of the COVID-19 nursing home data from the Centers for Medicare & Medicaid Services with a kernel density estimate on the diagonals.*


![image](https://cdn.discordapp.com/attachments/1217549861935775885/1218395840029397003/image.png?ex=6607828c&is=65f50d8c&hm=c214d1e209890daa345effd82a4eb7808e7674fa6d6ad524aadb4dc5a1600bde&)

 **Fig 3**: *Correlational heatmap between features of the COVID-19 nursing home data from the Centers for Medicare & Medicaid Services.*

### Pre-processing
We removed non-numerical columns (except for date which was integer converted), dropped the empty columns, and dropped all-time totals and columns already used in calculated columns. We also decided to drop the columns that are not contributing to our desired prediction (the number of cases per 1000 occupied beds inside of a nursing home during a given week), like the total amount in the dataset, or columns that are already included in calculated percentages/ratios. To deal with null/incomplete values, we removed rows with incomplete or null data. We will be MinMax scaling all the feature columns. For splitting the dataset, we had a ratio of 60:20:20 for training, testing, and validation respectively.

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
We chose our first model to be linear regression because we wanted to examine if there was a linear relationship between various factors of a nursing home and how many weekly COVID cases for every thousand occupied beds. We conducted it using sklearn’s LinearRegression model which uses ordinary least squares linear regression to get a closed form solution. The validation dataset was not considered in the training process.
```
reg = LinearRegression()
regmodel = reg.fit(X_train, y_train)
```
### Model 2: Simple Artificial Neural Network (Milestone 4):

For our next model, we used a Simple Artificial Neural Network composed of two hidden dense layers. We performed hyperparameter tuning to search for the best units per hidden layer, activation function, and learning rate. This created a neural network with 14 units per hidden layer, a sigmoid activation function, and a learning rate of 0.01, as these parameters resulted in the smallest validation loss. This model was trained for 20 epochs with a batch size of 100 using the Adam optimizer and mean squared error as loss on the training data, and then tested on the validation and test datasets.

![image](https://media.discordapp.net/attachments/1217549861935775885/1217635454044868668/image.png?ex=6604be62&is=65f24962&hm=bdf07425d2e10cacb76f8f518ff58641bf907cc8cd44a6772fa0fb87e802483c&=&format=webp&quality=lossless&width=1345&height=894)

 **Fig 4**: *Simple Artificial Neural Network Diagram, describing the structure and make up of our our second model*

```
keras.utils.set_random_seed(42)
def create_model(hp):
    model = Sequential()
    for i in range(2):
      model.add(
          Dense(
              units=hp.Choice('units', values=[7,14,28]),
              activation = hp.Choice('activation', values=['relu', 'sigmoid', 'tanh']),
          )
      )
    model.add(Dense(1, activation='linear'))
    learning_rate = hp.Float('lr', min_value=1e-3, max_value=1e-1, sampling='log', step = 10)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss = 'mse'
    )
    return model

tuner = RandomSearch(
  hypermodel=create_model,
  objective='val_loss',
  max_trials = 27,
  seed=0,
  executions_per_trial=1,
  overwrite=True,
  directory='model2_tuner_dir',
  project_name='model2'
)
tuner.search(X_train, y_train, epochs = 20, batch_size = 100, validation_data = (X_val, y_val))
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = create_model(best_hps)
history = best_model.fit(X_train, y_train, epochs=20, batch_size=100, validation_data=(X_val, y_val))
```

 ### Model #3: Deep Artificial Neural Network with Dropout

For our final model, we used a Deep Artificial Neural Network composed of four hidden dense layers and an additional dropout layer. Similar to the Simple Neural Network, we performed hyperparameter turning to search for the best units per dense hidden layer, activation function, and learning rate. This found a neural network with 28 units per dense hidden layer, a ‘ReLU’ activation function, and a learning rate of 0.001. We also included a dropout layer with a rate of 0.2 before the output layer. This model was trained for 40 epochs with a batch size of 100 using the Adam optimizer and mean squared error as loss on the training data, and then tested on the validation and test datasets.

![image](https://media.discordapp.net/attachments/1217549861935775885/1217638206573641848/image.png?ex=6604c0f3&is=65f24bf3&hm=eeef8e914558a05822703c2f5c194a95732a61a3b728d048ce17e00b5cab5449&=&format=webp&quality=lossless&width=1872&height=588)

 **Fig 5**: *Deep Artificial Neural Network Diagram, describing the structure and make up of our third model.*

```
keras.utils.set_random_seed(42)
def create_model3(hp):
    model = Sequential()
    for i in range(4):
      model.add(
          Dense(
              units=hp.Choice('units', values=[7,14,28]),
              activation = hp.Choice('activation', values=['relu', 'sigmoid', 'tanh']),
          )
      )
    model.add(Dropout(.2))
    model.add(Dense(1, activation='linear'))
    learning_rate = hp.Float('lr', min_value=1e-3, max_value=1e-1, sampling='log', step = 10)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss = 'mse'
    )
    return model

tuner_model3 = RandomSearch(
  hypermodel=create_model3,
  objective='val_loss',
  max_trials = 27,
  seed=0,
  executions_per_trial=1,
  overwrite=True,
  directory='model3_tuner_dir',
  project_name='model3'
)
tuner_model3.search(X_train, y_train, epochs = 40, batch_size = 100, validation_data = (X_val, y_val))
best_hps_model3 = tuner_model3.get_best_hyperparameters(num_trials=1)[0]
best_model3 = create_model3(best_hps_model3)
history_model3 = best_model3.fit(X_train, y_train, epochs=40, batch_size=100, validation_data=(X_val, y_val))
```
## Results

### Model 1: Linear Regression

Our training mean squared error for the linear regression model is 954.97, while our validation mean squared error is 926.85, and our test mean squared error is 951.46. There is no error tracking throughout the training process as the model performed a closed form solution.

### Model 2: Simple Artificial Neural Network

 We saw the following training and validation error throughout the epochs as seen in Fig.6. This resulted in a Training MSE of 802.51, a validation MSE of 767.20,  and a testing MSE of 807.91.

![image](https://cdn.discordapp.com/attachments/1197674736608628779/1217642064607580240/image.png?ex=6604c48a&is=65f24f8a&hm=be24418faa62dc8106b08be11081ca7ab647d9861128c704167c28a529c5b1c1&)

 **Fig 6**: *The Training and Validation Error over time (epochs) for our Simple Artificial Neural Network in predicting the cases per week per 1000 occupied beds in nursing homes*

### Model 3: Deep Artificial Neural Network with Dropout

We saw the following training and validation error throughout the epochs as seen in Fig. 7. This resulted in a training MSE of 793.34, a validation MSE of 763.74 and testing MSE of 801.45. 
 
![Model 3 Results](https://cdn.discordapp.com/attachments/1197674736608628779/1217642203820724224/image.png?ex=6604c4ac&is=65f24fac&hm=88b1acae1fe002eeafd1cf268daea10381d9cb01421878a4c450ae4c1b83ec92&)

**Fig 7**: *The Training and Validation Error over time (epochs) for our Deep Artificial Neural Network in predicting the cases per week per 1000 occupied beds in nursing homes*


### Final Results Summary

We compared all the three model’s performance results on the datasets in a histogram.

![Model Performance](https://cdn.discordapp.com/attachments/1217549861935775885/1217627186240950302/image.png?ex=6604b6af&is=65f241af&hm=2ea73d679df107fd75d0013eba20ea87e9af5ec54b1107b68c734c269c85f5f0&)

**Fig 8**: *The various mean squared errors (training, validation and testing) for each model (linear regression, simple artificial neural network, and deep neural network with dropout) in predicting the number of COVID-19 cases per week per 1000 occupied beds in nursing homes.*

## Discussion

### Idea
For this project, the motivation is the critical need to develop proactive strategies in the face of the pandemic's disproportionate impact on vulnerable populations and the possibility of achieving more effective resource allocation.

### Data Exploration and Preprocessing
As we gathered and preprocessed the data, we found out that a lot of attributes were corrupted, and some of them were not helpful for our desired prediction: the number of cases being treated inside a nursing home during a given week. As the corrupted data does not occupy a big percentage, we decided to drop corrupted and unrelated data. However, the data still demonstrated some points that are contrary to the expectation like the correlation between wklyResCasesPer1000 and percentStaffVaxUptoDate, percentResVaxUpToDate and wklyResCasesPer1000 are 0.0, but we still left them in in case they would be useful in more complex relationships. Even after we removed rows with incomplete or null data, we have plenty of data with over 50% of it complete for the remaining columns. We MinMax scaled all the features since the features/output were either multi-modal or are skewed making them non-normal, and they all have low p-values from the Shapiro test.

### Model 1: Linear Regression

We chose linear regression as our first model due to its simplicity and its ability to determine if there is a linear relationship between dependent variables and an independent variable. Looking at the results of the linear regression, our model is underfitted since the MSE for training and testing are both very high. This underfitting is explained by our dataset having several dependent variables with complex relationships between them and our independent variable, and the linear regression model is not best suited for those types of complex relationships. To improve this model we would likely have to compute columns with feature expansion or more realistically just use a more complex model.

### Model 2: Simple Artificial Neural Network
For the second model we built a Simple Artificial Neural Network. This model allowed us to consider the complex non-linear relationships that were not properly considered in the linear model. Using dense hidden layers will combat the underfitting that was seen in our linear regression model. 
 	When creating the model, we wanted to better grasp the design and decided to use tuning to find out the best hypermeter set. However, we decided to limit the amount of combinations to a relatively small number by limiting the number of choices of the nodes, learning rate, and activation function, as too many choices resulted in a really long runtime even under a GPU. After research we agreed on the number of nodes should at least be (input node + output node)/2 = (13+1)/2 = 7 nodes, thus we gave our tuner choice of 7, 14, 28. We also select choice of learning rate of 0.1, 0.01, 0.001 and common activation functions, making an combination of 3*3*3 = 27 (our maximum trials). After running the tuner we get the best-performing set of hyperparameters among the 27 sets, and the result is quite believable: the performance got better than model 1. Taking a look at the Training vs Validation error graph as seen in Fig 6, it seems to support that we are still underfitting since both training and validation error continue to be decreasing (though fairly slowly) during the training epochs. Both not training for enough epochs and the relative low complexity of the model could be contributing to the model underfitting. Compared to the first model though, it is definitely less underfit (more towards the center of the fitting graph) as we see lower testing and training errors. Our Artificial neural network may still not be sufficiently complex for the dataset and may not be able to learn the complex relationships between the features and our target. It still has a relatively high loss but did see improvement over the simpler linear regression model likely due to its additional complexity and ability to incorporate non-linearity thanks to the activation functions. The model was significantly slower than our first and more difficult to train which could be a concern if speed is an issue or if there is no access to a GPU. To improve, we could do more fine-tuned hyperparameter tuning and potentially add more epochs to the training to allow it to fit more to the dataset. Nevertheless, we were still disappointed in the overall loss and wanted to try a more complex model.

### Model 3: Deep Artificial Neural Network with Dropout
For our third model, we built a Deep Artificial Neural Network. Analyzing our simple neural network showed us that our model was still underfitted, meaning that we needed to increase the complexity of our model. We thought that adding more layers and training time will allow it to recognize more intricate patterns in the data which might help it avoid underfitting and generate a lower loss. This was our reason for doubling the hidden dense layers and epochs to allow the network to better model the complex relationships. We decided to keep similar hyperparameter options as our second model due to time constraints, resulting in a slightly better performance than before. Considering this model still failed to produce low error and has a nearly identical training/validation error graph as seen in Fig 7 compared to model 2, this model seems to also be underfit to our dataset. However, given this new failure we suspect that our data set might not be best for this task. Perhaps removing non-correlational columns and performing feature expansion, better cleaning our data, or performing a sort of under sampling for zero-valued targets would create more accurate results.

## Conclusion
Through each model, our goal was to increase the complexity to combat underfitting due to the nature of our features. With the increase in complexity, our model’s results became slightly more accurate with each step. Our mean squared error throughout each model never went below 750, revealing that it was difficult to take the minimal correlation found in pre-processing to an extremely accurate result. Nonetheless, the error improved upon complex models, reinstating the fact that our models were improving to properly predict COVID cases with the power of the model’s respective complexity. However, our progress did appear to be slowing down, so in the future perhaps taking another look at the features and cleaning of our dataset would be beneficial to improve errors significantly. Techniques such as under sampling or feature expansion could possibly lead to large improvements in accuracy. Furthermore, time and compute power did become a choke point for our team so having additional computational resources to train more complex models for more iterations would also likely improve performance. All in all, using machine learning to predict COVID case load seems to have promise, and although these were just preliminary explorations into possible solutions, a more in depth exploration could see success resulting in better medical resource allocation and outbreak planning for everyone.

## Collaboration section
We worked as a team with a fairly flat team structure.

Preston Ellis -- Project Coder
* Trained models locally
* Organized team Discord
* Coded for all milestones
* Wrote portions of and edited milestone READMEs
* Designed data preprocessing and model training
* Organized team Github
* Wrote portions of and edited the final report

Jackie Piepkorn -- Team Organizer
* Wrote the abstract
* Coded Milestone 2 and wrote a significant portion of the summary for the README
* Organized group meetings
* Reached out to group members via email and added everyone to Discord
* Wrote part of methods and made graphs for writeup

Steve Yin -- Project Coder
* Trained model locally
* Wrote portions of and edited the final report
* Coded for milestones

Dante Testini -- Project Analyst 
* Wrote significant portion of methods, including making diagrams, for the README
* Wrote significant part of discussion section of README
* Organized group meetings
* Active in group discussions

Ryan Martinez -- None
* Attended one meeting and wrote the Milestone 3 conclusion.

Sindhu Kothe -- None
* Attended one meeting
