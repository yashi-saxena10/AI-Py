from sklearn.model_selection import train_test_split
x = iris.iloc[:, :-1].values #last column values excluded
y = iris.iloc[:,   -1].values #last column value
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)  #Splitting the dataset into the Training set and Test set
