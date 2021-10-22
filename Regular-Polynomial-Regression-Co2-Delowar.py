#!/usr/bin/env python
# coding: utf-8

# # Polynomial Regression
# 
# 
# ## Objectives
# 
# To learn:
# 
# *   Use scikit-learn to implement Polynomial Regression
# *   Create a model, train it, test it and use the model
# 

# <h1>Table of contents</h1>
# 
# <div class="alert alert-block alert-info" style="margin-top: 20px">
#     <ol>
#         <li><a href="https://#download_data">Downloading Data</a></li>
#         <li><a href="https://#polynomial_regression">Polynomial regression</a></li>
#         <li><a href="https://#evaluation">Evaluation</a></li>
#         <li><a href="https://#practice">Practice</a></li>
#     </ol>
# </div>
# <br>
# <hr>
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


# Let's select some features that we want to use for regression.
# 

# In[ ]:


cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)


# Let's plot Emission values with respect to Engine size:
# 

# In[ ]:


plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# #### Creating train and test dataset
# 
# Train/Test Split involves splitting the dataset into training and testing sets respectively, which are mutually exclusive. After which, you train with the training set and test with the testing set.
# 

# In[ ]:


msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]


# <h2 id="polynomial_regression">Polynomial regression</h2>
# 

# Sometimes, the trend of data is not really linear, and looks curvy. In this case we can use Polynomial regression methods. In fact, many different regressions exist that can be used to fit whatever the dataset looks like, such as quadratic, cubic, and so on, and it can go on and on to infinite degrees.
# 
# In essence, we can call all of these, polynomial regression, where the relationship between the independent variable x and the dependent variable y is modeled as an nth degree polynomial in x. Lets say you want to have a polynomial regression (let's make 2 degree polynomial):
# 
# $$y = b + \theta\_1  x + \theta\_2 x^2$$
# 
# Now, the question is: how we can fit our data on this equation while we have only x values, such as **Engine Size**?
# Well, we can create a few additional features: 1, $x$, and $x^2$.
# 
# **PolynomialFeatures()** function in Scikit-learn library, drives a new feature sets from the original feature set. That is, a matrix will be generated consisting of all polynomial combinations of the features with degree less than or equal to the specified degree. For example, lets say the original feature set has only one feature, *ENGINESIZE*. Now, if we select the degree of the polynomial to be 2, then it generates 3 features, degree=0, degree=1 and degree=2:
# 

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])


poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)
train_x_poly


# **fit_transform** takes our x values, and output a list of our data raised from power of 0 to power of 2 (since we set the degree of our polynomial to 2).
# 
# The equation and the sample example is displayed below.
# 
# $$
# \begin{bmatrix}
# v\_1\\\\
# v\_2\\\\
# \vdots\\\\
# v_n
# \end{bmatrix}\longrightarrow \begin{bmatrix}
# \[ 1 & v\_1 & v\_1^2]\\\\
# \[ 1 & v\_2 & v\_2^2]\\\\
# \vdots & \vdots & \vdots\\\\
# \[ 1 & v_n & v_n^2]
# \end{bmatrix}
# $$
# 
# $$
# \begin{bmatrix}
# 2.\\\\
# 2.4\\\\
# 1.5\\\\
# \vdots
# \end{bmatrix} \longrightarrow \begin{bmatrix}
# \[ 1 & 2. & 4.]\\\\
# \[ 1 & 2.4 & 5.76]\\\\
# \[ 1 & 1.5 & 2.25]\\\\
# \vdots & \vdots & \vdots\\\\
# \end{bmatrix}
# $$
# 

# It looks like feature sets for multiple linear regression analysis, right? Yes. It Does.
# Indeed, Polynomial regression is a special case of linear regression, with the main idea of how do you select your features. Just consider replacing the  $x$ with $x\_1$, $x\_1^2$ with $x\_2$, and so on. Then the degree 2 equation would be turn into:
# 
# $$y = b + \theta\_1  x\_1 + \theta\_2 x\_2$$
# 
# Now, we can deal with it as 'linear regression' problem. Therefore, this polynomial regression is considered to be a special case of traditional multiple linear regression. So, you can use the same mechanism as linear regression to solve such a problems.
# 
# so we can use **LinearRegression()** function to solve it:
# 

# In[ ]:


clf = linear_model.LinearRegression()
train_y_ = clf.fit(train_x_poly, train_y)
# The coefficients
print ('Coefficients: ', clf.coef_)
print ('Intercept: ',clf.intercept_)


# As mentioned before, **Coefficient** and **Intercept** , are the parameters of the fit curvy line.
# Given that it is a typical multiple linear regression, with 3 parameters, and knowing that the parameters are the intercept and coefficients of hyperplane, sklearn has estimated them from our new set of feature sets. Lets plot it:
# 

# In[ ]:


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf.intercept_[0]+ clf.coef_[0][1]*XX+ clf.coef_[0][2]*np.power(XX, 2)
plt.plot(XX, yy, '-r' )
plt.xlabel("Engine size")
plt.ylabel("Emission")


# <h2 id="evaluation">Evaluation</h2>
# 

# In[ ]:


from sklearn.metrics import r2_score

test_x_poly = poly.fit_transform(test_x)
test_y_ = clf.predict(test_x_poly)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y,test_y_ ) )


# <h2 id="practice">Practice</h2>
# Try to use a polynomial regression with the dataset but this time with degree three (cubic). Does it result in better accuracy?
# 

# In[ ]:


# write your code here
poly3 = PolynomialFeatures(degree=3)
train_x_poly3 = poly3.fit_transform(train_x)
clf3 = linear_model.LinearRegression()
train_y3_ = clf3.fit(train_x_poly3, train_y)

# The coefficients
print ('Coefficients: ', clf3.coef_)
print ('Intercept: ',clf3.intercept_)
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf3.intercept_[0]+ clf3.coef_[0][1]*XX + clf3.coef_[0][2]*np.power(XX, 2) + clf3.coef_[0][3]*np.power(XX, 3)
plt.plot(XX, yy, '-r' )
plt.xlabel("Engine size")
plt.ylabel("Emission")
test_x_poly3 = poly3.fit_transform(test_x)
test_y3_ = clf3.predict(test_x_poly3)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y3_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y3_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y,test_y3_ ) )


# <details><summary>Click here for the solution</summary>
# 
# ```python
# poly3 = PolynomialFeatures(degree=3)
# train_x_poly3 = poly3.fit_transform(train_x)
# clf3 = linear_model.LinearRegression()
# train_y3_ = clf3.fit(train_x_poly3, train_y)
# 
# # The coefficients
# print ('Coefficients: ', clf3.coef_)
# print ('Intercept: ',clf3.intercept_)
# plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
# XX = np.arange(0.0, 10.0, 0.1)
# yy = clf3.intercept_[0]+ clf3.coef_[0][1]*XX + clf3.coef_[0][2]*np.power(XX, 2) + clf3.coef_[0][3]*np.power(XX, 3)
# plt.plot(XX, yy, '-r' )
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# test_x_poly3 = poly3.fit_transform(test_x)
# test_y3_ = clf3.predict(test_x_poly3)
# print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y3_ - test_y)))
# print("Residual sum of squares (MSE): %.2f" % np.mean((test_y3_ - test_y) ** 2))
# print("R2-score: %.2f" % r2_score(test_y,test_y3_ ) )
# 
# ```
# 
# </details>
# 

# 
