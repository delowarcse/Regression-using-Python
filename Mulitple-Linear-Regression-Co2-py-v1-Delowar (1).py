#!/usr/bin/env python
# coding: utf-8

# # Multiple Linear Regression¶
# 
# Objectives
# After completing this lab you will be able to:
# 
# Use scikit-learn to implement Multiple Linear Regression
# Create a model, train,test and use the model

# <h1>Table of contents</h1>
# 
# <div class="alert alert-block alert-info" style="margin-top: 20px">
#     <ol>
#         <li><a href="#understanding-data">Understanding the Data</a></li>
#         <li><a href="#reading_data">Reading the Data in</a></li>
#         <li><a href="#multiple_regression_model">Multiple Regression Model</a></li>
#         <li><a href="#prediction">Prediction</a></li>
#         <li><a href="#practice">Practice</a></li>
#     </ol>
# </div>
# <br>
# <hr>
# 

# ### Importing Needed packages
# 

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# <h2 id="understanding_data">Understanding the Data</h2>
# 
# ### `FuelConsumption.csv`:
# 
# We have downloaded a fuel consumption dataset, **`FuelConsumption.csv`**, which contains model-specific fuel consumption ratings and estimated carbon dioxide emissions for new light-duty vehicles for retail sale in Canada. [Dataset source](http://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64?cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork-20718538&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork-20718538&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork-20718538&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork-20718538&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ)
# 
# -   **MODELYEAR** e.g. 2014
# -   **MAKE** e.g. Acura
# -   **MODEL** e.g. ILX
# -   **VEHICLE CLASS** e.g. SUV
# -   **ENGINE SIZE** e.g. 4.7
# -   **CYLINDERS** e.g 6
# -   **TRANSMISSION** e.g. A6
# -   **FUELTYPE** e.g. z
# -   **FUEL CONSUMPTION in CITY(L/100 km)** e.g. 9.9
# -   **FUEL CONSUMPTION in HWY (L/100 km)** e.g. 8.9
# -   **FUEL CONSUMPTION COMB (L/100 km)** e.g. 9.2
# -   **CO2 EMISSIONS (g/km)** e.g. 182   --> low --> 0
# 

# <h2 id="reading_data">Reading the data in</h2>
# 

# In[3]:


df = pd.read_csv("FuelConsumption.csv")

# take a look at the dataset
df.head()


# Lets select some features that we want to use for regression.
# 

# In[4]:


cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)


# Lets plot Emission values with respect to Engine size:
# 

# In[5]:


plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# #### Creating train and test dataset
# 
# Train/Test Split involves splitting the dataset into training and testing sets respectively, which are mutually exclusive. After which, you train with the training set and test with the testing set. 
# This will provide a more accurate evaluation on out-of-sample accuracy because the testing dataset is not part of the dataset that have been used to train the data. It is more realistic for real world problems.
# 
# This means that we know the outcome of each data point in this dataset, making it great to test with! And since this data has not been used to train the model, the model has no knowledge of the outcome of these data points. So, in essence, it’s truly an out-of-sample testing.
# 

# In[6]:


msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]


# #### Train data distribution
# 

# In[7]:


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# <h2 id="multiple_regression_model">Multiple Regression Model</h2>
# 

# In reality, there are multiple variables that predict the Co2emission. When more than one independent variable is present, the process is called multiple linear regression. For example, predicting co2emission using FUELCONSUMPTION_COMB, EngineSize and Cylinders of cars. The good thing here is that Multiple linear regression is the extension of simple linear regression model.
# 

# In[8]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x, y)
# The coefficients
print ('Coefficients: ', regr.coef_)


# As mentioned before, **Coefficient** and **Intercept** , are the parameters of the fit line. 
# Given that it is a multiple linear regression, with 3 parameters, and knowing that the parameters are the intercept and coefficients of hyperplane, sklearn can estimate them from our data. Scikit-learn uses plain Ordinary Least Squares method to solve this problem.
# 
# #### Ordinary Least Squares (OLS)
# 
# OLS is a method for estimating the unknown parameters in a linear regression model. OLS chooses the parameters of a linear function of a set of explanatory variables by minimizing the sum of the squares of the differences between the target dependent variable and those predicted by the linear function. In other words, it tries to minimizes the sum of squared errors (SSE) or mean squared error (MSE) between the target variable (y) and our predicted output ($\hat{y}$) over all samples in the dataset.
# 
# OLS can find the best parameters using of the following methods:
# 
# ```
# - Solving the model parameters analytically using closed-form equations
# - Using an optimization algorithm (Gradient Descent, Stochastic Gradient Descent, Newton’s Method, etc.)
# ```
# 

# <h2 id="prediction">Prediction</h2>
# 

# In[9]:


y_hat= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squares: %.2f"
      % np.mean((y_hat - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))


# **explained variance regression score:**  
# If $\hat{y}$ is the estimated target output, y the corresponding (correct) target output, and Var is Variance, the square of the standard deviation, then the explained variance is estimated as follow:
# 
# $\texttt{explainedVariance}(y, \hat{y}) = 1 - \frac{Var{ y - \hat{y}}}{Var{y}}$  
# The best possible score is 1.0, lower values are worse.
# 

# <h2 id="practice">Practice</h2>
# Try to use a multiple linear regression with the same dataset but this time use __FUEL CONSUMPTION in CITY__ and 
# __FUEL CONSUMPTION in HWY__ instead of FUELCONSUMPTION_COMB. Does it result in better accuracy?
# 

# In[12]:


# write your code here
regr1 = linear_model.LinearRegression()
x1 = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y1 = np.asanyarray(train[['CO2EMISSIONS']])
regr1.fit (x1, y1)
# The coefficients
print ('Coefficients: ', regr1.coef_)

y_hat1= regr1.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
x1 = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y1 = np.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squares: %.2f"
      % np.mean((y_hat1 - y1) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr1.score(x1, y1))


# <details><summary>Click here for the solution</summary>
# 
# ```python
# regr = linear_model.LinearRegression()
# x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
# y = np.asanyarray(train[['CO2EMISSIONS']])
# regr.fit (x, y)
# print ('Coefficients: ', regr.coef_)
# y_= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
# x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
# y = np.asanyarray(test[['CO2EMISSIONS']])
# print("Residual sum of squares: %.2f"% np.mean((y_ - y) ** 2))
# print('Variance score: %.2f' % regr.score(x, y))
# 
# ```
# 
# </details>
# 
