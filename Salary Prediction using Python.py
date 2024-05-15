#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

data = pd.read_csv("C:/Users/student/Desktop/Salary_data.csv")

# figure = px.scatter(data_frame = data, 
#                     x="Salary",
#                     y="YearsExperience", 
#                     size="YearsExperience", 
#                     trendline="ols")
# figure.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x = np.asanyarray(data[["YearsExperience"]])
y = np.asanyarray(data[["Salary"]])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)

a = float(input("Years of Experience: "))
features = np.array([[a]])
predicted_salary = model.predict(features)[0][0] 
predicted_salary_rounded = round(predicted_salary, 1) 
print("Predicted Salary =", predicted_salary_rounded)

