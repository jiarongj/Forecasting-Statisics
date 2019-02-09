from __future__ import print_function
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt

# Artificial data
nsample = 50
sig = 0.25
x1 = np.linspace(0,20,nsample)
X = np.column_stack((x1, np.sin(x1), (x1-5)**2))    # add the two values based on x1 to each x1 value to constitute 3 member for each elemens
print(X)
X = sm.add_constant(X)    # Adds a column of ones to an array
# # print(X)
beta = [5., 0.5, 0.5, -0.02]
y_true = np.dot(X, beta)    # [50x4] dot product with [1x4] vector
# print(y_true)
y = y_true + sig * np.random.normal(size = nsample)  # add value 'delta' to each of 50
print(y)
olsmod = sm.OLS(y,X)
olsres = olsmod.fit()
print(olsres.summary())
ypred = olsres.predict(X)
print(ypred)
print('Parameters:', olsres.params)
print('R2:', olsres.rsquared)

# Create an extra sample set of explanatory variables Xnew to predict and plot
xln = np.linspace(20.5, 25, 10)
Xnew = np.column_stack((xln, np.sin(xln), (xln-5)**2))
Xnew = sm.add_constant(Xnew)
ynewpred = olsres.predict(Xnew)
print(ynewpred)

# Plot to test the prediction accuracy
fig, ax = plt.subplots()   # prepare a blank 'fig' with name 'ax'
ax.plot(x1, y, 'o', label = "Data")
ax.plot(x1, y_true, 'b-', label ="True")
ax.plot(np.hstack((x1, xln)), np.hstack((ypred, ynewpred)), 'r', label = "OLS prediction") # Overlap the two (legends)sets of x and y values
ax.legend(loc = "best");
plt.show()

# Predicting with Formulas(can make both estimation and prediction a lot easier)
data = {"x1":x1, "y":y}
res = ols("y ~ x1 + np.sin(x1) + I((x1-5)**2)", data = data).fit()  # Use I to indicate the use of identity transform
print(res.params)
print(res.predict(exog=dict(x1=xln)))
# Use the ols model to fit the data, and then use the results to predict the other data group

