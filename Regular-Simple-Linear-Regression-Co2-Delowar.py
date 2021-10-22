#!/usr/bin/env python
# coding: utf-8

# # Simple Linear Regression
# 
# ## Objectives
# 
# To learn:
# 
# *   Use scikit-learn to implement simple Linear Regression
# *   Create a model, train it, test it and use the model
# 

# ### Importing Needed packages
# 

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Understanding the Data
# 
# ### `FuelConsumption.csv`:
# 
# We have downloaded a fuel consumption dataset, **`FuelConsumption.csv`**, which contains model-specific fuel consumption ratings and estimated carbon dioxide emissions for new light-duty vehicles for retail sale in Canada. [Dataset source](http://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2021-01-01)
# 
# *   **MODELYEAR** e.g. 2014
# *   **MAKE** e.g. Acura
# *   **MODEL** e.g. ILX
# *   **VEHICLE CLASS** e.g. SUV
# *   **ENGINE SIZE** e.g. 4.7
# *   **CYLINDERS** e.g 6
# *   **TRANSMISSION** e.g. A6
# *   **FUEL CONSUMPTION in CITY(L/100 km)** e.g. 9.9
# *   **FUEL CONSUMPTION in HWY (L/100 km)** e.g. 8.9
# *   **FUEL CONSUMPTION COMB (L/100 km)** e.g. 9.2
# *   **CO2 EMISSIONS (g/km)** e.g. 182   --> low --> 0
# 

# ## Reading the data in
# 

# In[ ]:


df = pd.read_csv("FuelConsumption.csv")

# take a look at the dataset
df.head()


# ### Data Exploration
# 
# Let's first have a descriptive exploration on our data.
# 

# In[ ]:


# summarize the data
df.describe()


# Let's select some features to explore more.
# 

# In[ ]:


cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)


# We can plot each of these features:
# 

# In[ ]:


viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()


# Now, let's plot each of these features against the Emission, to see how linear their relationship is:
# 

# In[ ]:


plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()


# In[ ]:


plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# ## Practice
# 
# Plot **CYLINDER** vs the Emission, to see how linear is their relationship is:
# 

# In[ ]:


# write your code here

plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Cylinders")
plt.ylabel("Emission")
plt.show(


# <details><summary>Click here for the solution</summary>
# 
# ```python
# plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='blue')
# plt.xlabel("Cylinders")
# plt.ylabel("Emission")
# plt.show()
# 
# ```
# 
# </details>
# 

# #### Creating train and test dataset
# 
# Train/Test Split involves splitting the dataset into training and testing sets that are mutually exclusive. After which, you train with the training set and test with the testing set.
# This will provide a more accurate evaluation on out-of-sample accuracy because the testing dataset is not part of the dataset that have been used to train the model. Therefore, it gives us a better understanding of how well our model generalizes on new data.
# 
# This means that we know the outcome of each data point in the testing dataset, making it great to test with! Since this data has not been used to train the model, the model has no knowledge of the outcome of these data points. So, in essence, it is truly an out-of-sample testing.
# 
# Let's split our dataset into train and test sets. 80% of the entire dataset will be used for training and 20% for testing. We create a mask to select random rows using **np.random.rand()** function:
# 

# In[ ]:


msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]


# ### Simple Regression Model
# 
# Linear Regression fits a linear model with coefficients B = (B1, ..., Bn) to minimize the 'residual sum of squares' between the actual value y in the dataset, and the predicted value yhat using linear approximation.
# 

# #### Train data distribution
# 

# In[ ]:


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# #### Modeling
# 
# Using sklearn package to model data.
# 

# In[ ]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)


# As mentioned before, **Coefficient** and **Intercept** in the simple linear regression, are the parameters of the fit line.
# Given that it is a simple linear regression, with only 2 parameters, and knowing that the parameters are the intercept and slope of the line, sklearn can estimate them directly from our data.
# Notice that all of the data must be available to traverse and calculate the parameters.
# 

# #### Plot outputs
# 

# We can plot the fit line over the data:
# 

# In[ ]:


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")


# #### Evaluation
# 
# We compare the actual values and predicted values to calculate the accuracy of a regression model. Evaluation metrics provide a key role in the development of a model, as it provides insight to areas that require improvement.
# 
# There are different model evaluation metrics, lets use MSE here to calculate the accuracy of our model based on the test set:
# 
# *   Mean Absolute Error: It is the mean of the absolute value of the errors. This is the easiest of the metrics to understand since it’s just average error.
# 
# *   Mean Squared Error (MSE): Mean Squared Error (MSE) is the mean of the squared error. It’s more popular than Mean Absolute Error because the focus is geared more towards large errors. This is due to the squared term exponentially increasing larger errors in comparison to smaller ones.
# 
# *   Root Mean Squared Error (RMSE).
# 
# *   R-squared is not an error, but rather a popular metric to measure the performance of your regression model. It represents how close the data points are to the fitted regression line. The higher the R-squared value, the better the model fits your data. The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
# 

# In[ ]:


from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )

