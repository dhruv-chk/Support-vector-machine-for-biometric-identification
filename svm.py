# Support Vector Machine (SVM)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("biometricset.csv")
X = dataset.iloc[:, [0, 5]].values
y = dataset.iloc[:, 6].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
count=0
for i in range(len(y_test)):
    if y_test[i]==0:
        count+=1
print("Number of 0="+str(count))
count=0
for i in range(len(y_test)):
    if y_test[i]==1:
        count+=1
print("Number of 1="+str(count))
count=0
for i in range(len(y_test)):
    if y_test[i]==2:
        count+=1
print("Number of 2="+str(count))
count=0
for i in range(len(y_test)):
    if y_test[i]==3:
        count+=1
print("Number of 3="+str(count))
count=0
for i in range(len(y_test)):
    if y_test[i]==4:
        count+=1
print("Number of 4="+str(count))
count=0
for i in range(len(y_test)):
    if y_test[i]==5:
        count+=1
print("Number of 5="+str(count))
count=0
for i in range(len(y_test)):
    if y_test[i]==6:
        count+=1
print("Number of 6="+str(count))
count=0
for i in range(len(y_test)):
    if y_test[i]==7:
        count+=1
print("Number of 7="+str(count))
count=0
for i in range(len(y_test)):
    if y_test[i]==8:
        count+=1
print("Number of 8="+str(count))
count=0
for i in range(len(y_test)):
    if y_test[i]==9:
        count+=1
print("Number of 9="+str(count))
count=0
for i in range(len(y_test)):
    if y_test[i]==10:
        count+=1
print("Number of 10="+str(count))

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()