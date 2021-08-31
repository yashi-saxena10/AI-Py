from sklearn.tree import   DecisionTreeClassifier
from sklearn.metrics import accuracy_score
classifier   = DecisionTreeClassifier()
classifier.fit(x_train,   y_train) #training the classifier
y_pred   = classifier.predict(x_test) #making precdictions
print(classification_report(y_test,   y_pred)) #Summary of the predictions made by the classifier
print(confusion_matrix(y_test, y_pred)) #to evaluate the quality of the output
print('accuracy   is',accuracy_score(y_pred,y_test)) #Accuracy score

#Heatmap for confusion matrix
import seaborn as sns
cm  = confusion_matrix(y_test, y_pred) #Transform to df
cm_df = pd.DataFrame(cm,index = ['setosa','versicolor','virginica'], columns = ['setosa','versicolor','virginca'])
plt.figure(figsize=(5.5,4))
sns.heatmap(cm_df,   annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
