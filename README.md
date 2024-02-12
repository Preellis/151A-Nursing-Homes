# CSE 151A Nursing Homes

## Project Abstract
Nursing homes have a variety of factors that contribute to the number of COVID-19 cases making predicting case loads difficult. Using data science and machine learning methods, we aim to find strategies to predict nursing home COVID-19 cases. Using nursing home data from US Centers for Medicare and Medicaid Services, we want to predict the number of cases being treated inside of a nursing home during a given week based on whether the home passed the quality assurance test, number of total beds in the home, the number of residents who are up to date on COVID 19 vaccinations 14 days or more before their positive test, and the number of staff that week that have tested positive for COVID 19. To make these predictions, we will implement both linear regression and a neural network. Linear regression relies on a linear relationship between independent and dependent variables to predict unknown values, while neural networks may be a better choice for more complex datasets with a non-linear relationship between variables. We will compare the methods based on their respective accuracy, precision, and recall to better understand which machine learning model best fits the proposed problem. Our results will help us better understand the containment of the virus and may be useful in making healthcare decisions for the well-being of nursing home inhabitants.

## How We Will Preprocess Data
We will implement normalization and standardization of the data, as well as remove rows that have null values.

<a target="_blank" href="https://colab.research.google.com/github/Preellis/151A-Nursing-Homes">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
