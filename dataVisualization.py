import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
iris =   pd.read_csv("Iris.csv") #Iris.csv is now a pandas dataframe
print(iris.head()) #prints first 5 values
print(iris.describe()) #prints some basic statistical details like percentile,mean, std etc of the data frame
#Scatter plot
iris.plot(kind="scatter", x="SepalLengthCm",   y="SepalWidthCm")
plt.show()
#Andrews curves plot
from pandas.plotting import andrews_curves
andrews_curves(iris.drop("Id", axis=1), "Species")
plt.show()
