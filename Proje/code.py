import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from mpl_toolkits.mplot3d import axes3d
import statistics
from scipy.stats import zscore

TEST_SIZE = 0.3

def plot_ROCs(test, preds, labels):
    plt.figure(figsize=(5, 5), dpi=100)
    for i in range(len(labels)):
        fpr, tpr, threshold3 = roc_curve(test, preds[i])
        auc_tree = auc(fpr, tpr)
        plt.plot(fpr, tpr, marker='.', linestyle='--', label=labels[i] + ' (auc = %0.3f)' % auc_tree)
    plt.title("ROC Curves")
    plt.xlabel('False Positive Rate -->')
    plt.ylabel('True Positive Rate -->')
    plt.legend()
    plt.show()

def plot_ROC(test, pred, label):
    fpr, tpr, threshold3 = roc_curve(test, pred)
    auc_tree = auc(fpr, tpr)

    plt.figure(figsize=(5, 5), dpi=100)
    plt.plot(fpr, tpr, marker='.', label=label + ' (auc = %0.3f)' % auc_tree)

    plt.title(label + " ROC Curve")
    plt.xlabel('False Positive Rate -->')
    plt.ylabel('True Positive Rate -->')
    plt.legend()
    plt.show()

def plot_SVM(data, row_1, row_2, model, title):
    #dataset, row1st, row2nd, model, header
    X_set, x_, y_set, y_ = train_test_split(
        data.iloc[:, row_1: row_2].values,
        data.iloc[:, -1].values,
        test_size = TEST_SIZE,
        random_state = 42
    )

    X_set = StandardScaler().fit_transform(X_set)
    mmodel = model.fit(X_set, y_set)

    X1, X2 = np.meshgrid(
        np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1, 0.01),
        np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, 0.01)
    )
    
    plt.contourf(
        X1, X2,
        mmodel.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
        alpha = 0.75,
        cmap = ListedColormap(('blue', 'green'))
    )

    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())

    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(
            X_set[y_set == j, 0],
            X_set[y_set == j, 1],
            c = ListedColormap(('yellow', 'green'))(i), label = j
        )
    plt.title(title + " Support Vector Machine (SVM) Graph")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

def plot_confusion_matrix(Ytest, Ypred, label):
    confusionMatrix = confusion_matrix(Ytest, Ypred)
    f, ax = plt.subplots(figsize=(5, 5))
    
    sns.heatmap(confusionMatrix, annot=True, linewidth=0.7, linecolor='cyan', fmt='g', ax=ax, cmap="YlGnBu")
    plt.title(label + " Confusion Matrix")
    plt.xlabel('Y Tahmin')
    plt.ylabel('Y Test')
    plt.show()

def lenOfList(list):
    newList = []
    for i in range(1, len(list) + 1):
        newList.append(str(i))
    return newList

def log_MSEs(test, preds, labels):
    mse = []
    for i in range(len(preds)):
        mse.append(mean_squared_error(test, preds[i]))
        print(labels[i] + " MSE: " + str(mse))
    fig, axes = plt.subplots(figsize=(35,10), dpi=100)
    plt.bar(labels, height=mse, color=['red', 'gray', 'blue', 'orange', 'purple', 'yellow', 'pink'])
    plt.title("MSE Values")
    plt.show()

def cross_valid_graph(model, X, y, model_name, color='gray', cv=30):
    scores = cross_val_score(model, X, y, cv=cv)
    fig, axes = plt.subplots(figsize=(35,10), dpi=100)
    sns.set(font_scale=1.5)
    plt.bar(range(1, len(scores) + 1), height=scores, color=color)
    plt.title(model_name + " Cross Validation\nAccuracy: " + str(scores.mean()) + "\nStandard Deviation: " + str(scores.std()))
    plt.show()
    return scores


# Dataseti tanımladık
data = pd.read_csv("./Proje/winequality-red.csv")
data.corr()['quality'].sort_values(ascending=False)

print(data.mean())
print(data.info())
print(data.head())
print(data.describe()) # minimum, maksimum, ortalama, standart sapma, medyan değerleri
print(data.isnull().any()) # özniteliklerde eksik olup olmadığını kontrol ettik
print(data.isnull().sum())
print(data.all())
print(data.dtypes)
print(data.columns)
print(data.shape)
print(data.value_counts())

# Dataset ile ilgili verilerin görselleştirilmesi
sns.heatmap(data.corr(),annot=True, cmap='Blues')
sns.pairplot(data, diag_kind='kde')
plt.show()

X=zscore(data.drop(['quality'],axis=1))
y=data['quality']
#X=data.iloc[:,0:-1]
#y=data.iloc[:,-1] 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=TEST_SIZE,random_state=42)
preds = []
cross_scores = []
algorithms = [
    "Nearest Neighbor Classification",
    "C-Support Vector Classification",
    "Logistic Regression",
    "Gaussian Naive Bayes",
    "Decision Tree Classifier",
    "Bagging Classifier",
    "RandomForestClassifier"
]

sns.set(font_scale=1.5)
# NEAREST NEIGHBOR CLASSIFICATION
knnModel = KNeighborsClassifier(n_neighbors=8, metric='manhattan')
knnModel.fit(X_train, y_train)
Ypred_knn = knnModel.predict(X_test)
preds.append(Ypred_knn)

plot_confusion_matrix(y_test, Ypred_knn, algorithms[0])
scores = [[algorithms[0], knnModel.score(X_test, y_test)]]
# plot_ROC(y_test, Ypred_knn, algorithms[0])

# SUPPORT VECTOR CLASSIFICATION
svcModel = SVC(C=10, random_state=42)
svcModel.fit(X_train, y_train)
Ypred_svc = svcModel.predict(X_test)
preds.append(Ypred_svc)

scores.append([algorithms[1], svcModel.score(X_test, y_test)])
plot_confusion_matrix(y_test, Ypred_svc, algorithms[1])
plot_SVM(data, 0, 2, svcModel, algorithms[1])
# plot_ROC(y_test, Ypred_svc, algorithms[1])

# LOGISTIC REGRESSION
logisticModel = LogisticRegression(C=10, random_state=42)
logisticModel.fit(X_train, y_train)
Ypred_logistic = logisticModel.predict(X_test)
preds.append(Ypred_logistic)

scores.append([algorithms[2], logisticModel.score(X_test, y_test)])
plot_confusion_matrix(y_test, Ypred_logistic, algorithms[2])
# plot_ROC(y_test, Ypred_logistic, algorithms[2])

# NAIVE BAYES CLASSIFIER
naiveBayesModel = GaussianNB()
naiveBayesModel.fit(X_train, y_train)
Ypred_naiveBayes = naiveBayesModel.predict(X_test)
preds.append(Ypred_naiveBayes)

scores.append([algorithms[3], naiveBayesModel.score(X_test,y_test)])
plot_confusion_matrix(y_test, Ypred_naiveBayes, algorithms[3])
# plot_ROC(y_test, Ypred_naiveBayes, algorithms[3])

# DECISION TREE CLASSIFIER
decisionTreeModel = DecisionTreeClassifier(random_state=42)
decisionTreeModel.fit(X_train, y_train)
Ypred_decisionTree = decisionTreeModel.predict(X_test)
preds.append(Ypred_decisionTree)

scores.append([algorithms[4], decisionTreeModel.score(X_test, y_test)])
plot_confusion_matrix(y_test, Ypred_decisionTree, algorithms[4])
# plot_ROC(y_test, Ypred_decisionTree, algorithms[4])

# BaggingClassifier CLASSIFIER
baggingModel = BaggingClassifier(base_estimator=decisionTreeModel, n_estimators=10, random_state=17)
baggingModel.fit(X_train, y_train)
Ypred_bagging = baggingModel.predict(X_test)
preds.append(Ypred_bagging)

scores.append([algorithms[5], baggingModel.score(X_test, y_test)])
plot_confusion_matrix(y_test, Ypred_bagging, algorithms[5])
# plot_ROC(y_test, Ypred_bagging, algorithms[5])

# RANDOM FOREST TREE CLASSIFIER
randomForestTreeModel = RandomForestClassifier(random_state=42)
randomForestTreeModel.fit(X_train, y_train)
Ypred_randomForestTree = randomForestTreeModel.predict(X_test)
preds.append(Ypred_randomForestTree)

cross_scores.append(cross_valid_graph(randomForestTreeModel, X, y, algorithms[6], 'pink'))
cross_scores.append(cross_valid_graph(logisticModel, X, y, algorithms[2], 'blue'))
cross_scores.append(cross_valid_graph(svcModel, X, y, algorithms[1], 'gray'))
cross_scores.append(cross_valid_graph(naiveBayesModel, X, y, algorithms[3], 'orange'))
cross_scores.append(cross_valid_graph(baggingModel, X, y, algorithms[5], 'yellow'))
cross_scores.append(cross_valid_graph(knnModel, X, y, algorithms[0], 'red'))
cross_scores.append(cross_valid_graph(decisionTreeModel, X, y, algorithms[4], 'purple'))

scores.append([algorithms[6], randomForestTreeModel.score(X_test, y_test)])
plot_confusion_matrix(y_test, Ypred_randomForestTree, algorithms[6])
plot_ROC(y_test, Ypred_randomForestTree, algorithms[6])

plot_ROCs(y_test, preds, algorithms)
sns.set(font_scale=1.0)
log_MSEs(y_test, preds, algorithms)

a = []
b = []

for i in range(len(algorithms)):
    print(algorithms[i] + str(scores[i]))

for i in scores:
    a.append(i[0])
    b.append(i[1])

# İkiye ayırdığımız sonuç verilerini grafiğe döktük.
fig, axes = plt.subplots(figsize=(35,10), dpi=100)
plt.bar(a, height=b, color=['red', 'gray', 'blue', 'orange', 'purple', 'yellow', 'pink'])
plt.title('Kırmızı Şarap Kalitesi')
plt.show()

for cross in cross_scores:
    print("Cross: " + str(statistics.mean(cross)))

cross_means = []
for i in range(len(cross_scores)):
    cross_scores[i].sort()
    cross_means.append(statistics.mean(cross_scores[i]))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X = [
    "Nearest Neighbor Classification",
    "Decision Tree Classifier",
    "Bagging Classifier",
    "C-Support Vector Classification",
    "Gaussian Naive Bayes",
    "Logistic Regression",
    "RandomForestClassifier"
]

x = np.array([[i] * 30 for i in range(1, 8)]).ravel() #algoritma çalışma sayısı
y = np.array([i for i in range(1, 31)] * len(X))
z = np.zeros(len(X)*30)

dx = np.ones(len(X)*30) # length along x-axis of each bar
dy = np.ones(len(X)*30) # length along y-axis of each bar
dz = np.array(cross_scores).ravel() # length along z-axis of each bar (height)

xs = np.random.rand(100)
ys = np.random.rand(100)
zs = np.random.rand(100)

from matplotlib import cm
from matplotlib.colors import Normalize
cmap = cm.get_cmap('plasma')
norm = Normalize(vmin=min(dz), vmax=max(dz))
colors = cmap(norm(dz))
sc = cm.ScalarMappable(cmap=cmap,norm=norm)
sc.set_array([])
plt.colorbar(sc)

ax.bar3d(x, y, z, dx, dy, dz, color=colors, zsort='average')
ax.set_ylabel('Cross-Validation Çalışmaları')
ax.set_zlabel('Cross-Validation Çıktıları')

plt.show()