from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

data = load_breast_cancer()
X = data.data
Y=data.target

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

clf1 = DecisionTreeClassifier()
clf1.fit(x_train, y_train)

clf2 = RandomForestClassifier()
clf2.fit(x_train, y_train)

print(f'DTC: {clf1.score(x_test,y_test)}')
print(f'RFC: {clf2.score(x_test,y_test)}')