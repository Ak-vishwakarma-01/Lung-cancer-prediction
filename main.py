import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
""" use of train_test_split
data set - 100 record
split = train and test
  split -  60(to train)
           40(to test )
"""
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn import tree
import math

print("Datasset:")
dataset = pd.read_csv('lung_cancer.csv')
# print(dataset) 
print(len(dataset)) # it wil print lenght of the data
print(dataset.head()) #it will print somedata just to see that what are the datas

scatter_matrix(dataset)
pyplot.show()
 
A = dataset[dataset.Result == 1] #when the result is 1 then it will taking the vlaue A 
B = dataset[dataset.Result == 0] #when the result is o then it will taking the value B

plt.scatter(A.Age, A.Smokes, color="Black", label="1", alpha=0.4)  #1 is for who is having cancer 
plt.scatter(B.Age, B.Smokes, color="Blue", label="0", alpha=0.4)   #0 is for who is having no cancer
plt.xlabel("Age")
plt.ylabel("Smokes")
plt.legend()
plt.title("Smokes vs Age")
plt.show()


plt.scatter(A.Age, A.Alkhol, color="Black", label="1",alpha=0.4)
plt.scatter(B.Age, B.Alkhol, color="Blue", label="0",alpha=0.4)
plt.xlabel("Age")
plt.ylabel("Alkhol")
plt.legend()
plt.title("Alkhol vs Age")
plt.show()

plt.scatter(A.Smokes, A.Alkhol, color="Black", label="1",alpha=0.4)
plt.scatter(B.Smokes, B.Alkhol, color="Blue", label="0",alpha=0.4)
plt.xlabel("Smokes")
plt.ylabel("Alkhol")
plt.legend()
plt.title("Alkhol vs Smokes")
plt.show()


#spliting dataset
x = dataset.iloc[:,3:5]
y = dataset.iloc[:,6]
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=0, test_size=0.2)


# feature Scaling

sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train) #fitting the model
x_test = sc_x.transform(x_test)


print('-----------------------****Using KNN Algorithm***------------------------------')
a = math.sqrt(len(y_train))
print(a)


#defining a model - KNN
classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric = 'eculidien')

#fit model
classifier.fit(x_train, y_train)


#predict the test result
y_pred = classifier.predict(x_test)
print(y_pred)


#Evaluate model
#confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(cm)
print("In Confusion Matrix:-----")
print("Position 1.1 shows the patients that don't have Cancer, In this case = 8")
print("Position 1.2 shows the number of patients that have higher risk of Cancer, In this case = 1")
print("Postion 2.1 shows the Incorrect Value, In this case = 2")
print("Position 2.2 shows the correct number of patients that have Cancer, In this case = 2")

print('F1 Score : ',(f1_score(y_test, y_pred))*100)
print('ACCURACY :',(accuracy_score(y_test, y_pred))*100)



#using decision treee
print('-----------------------****Using Decision Tree Algorithm***------------------------------')
c=tree.DecisionTreeClassifier()
c.fit(x_train, y_train)
accu_train = np.sum(c.predict(x_train)==y_train) / float(y_train.size)
accu_test = np.sum(c.predict(x_test)==y_test) / float(y_test.size)
print('Classsification accuracy on train',(accu_train)*100)
print('Classification accuracy on test', (accu_test)*100)
