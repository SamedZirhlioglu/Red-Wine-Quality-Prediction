import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Değişkenlerin DEFINE edilmesi
TEST_SIZE = 0.3
RANDOM_SEED = 42
DATA_PATH = r'C:\Users\Mehtap\Desktop\classification\data.csv'
DATA_PATH = 'winequality-red.csv'

# Confusion Matrix grafiğini oluşturan fonksiyon
def plot_confusion_matrix(Ytest, Ypred, label):
    confusionMatrix = confusion_matrix(Ytest, Ypred)
    f, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(confusionMatrix, annot=True, fmt='g', ax=ax)
    plt.title(label + " Confusion Matrix")
    plt.xlabel('Predict')
    plt.ylabel('Valid')
    plt.show()

# Cross-Validation çıktılarının grafiğini oluşturan fonksiyon
def plot_cross_valid(model_name, cross_scores, color='gray'):
    fig, axes = plt.subplots(figsize=(35, 10), dpi=100)
    plt.bar(range(len(cross_scores)), height=cross_scores, color=color)
    plt.title(model_name + " Cross Validation\nAccuracy (mean): " + str(cross_scores.mean()) + "\nStandard Deviation: " + str(cross_scores.std()))
    plt.show()

# Model tahminlerinin ROC grafiğini oluşturan fonksiyon
def plot_ROC(test, preds):
    plt.figure(figsize=(5, 5), dpi=100)
    for pred in preds:
        fpr, tpr, threshold3 = roc_curve(test, pred[1])
        auc_tree = auc(fpr, tpr)
        plt.plot(fpr, tpr, marker='.', linestyle='--', label=pred[0] + ' (AUC = %0.3f)' % auc_tree)
    plt.title("ROC Curves")
    plt.xlabel('False Positive Rate -->')
    plt.ylabel('True Positive Rate -->')
    plt.legend()
    plt.show()

# Model skorlarının grafiğini oluşturan fonksiyon
def plot_model_scores(results, colors = ['gray', 'gray']):
    for i in range(len(results)):
        title = results[i][0]
        scores = [results[i][1], results[i][2]]
        plt.subplot(1, 2, i + 1)
        plt.bar(['Accuracy', 'MSE'], height=scores, color=colors[i])
        plt.title(title)
    plt.show()

# Veri setinin DataFrame olarak import edilmesi
df = pd.read_csv(DATA_PATH)

# Import edilen DataFrame'in X-y (öznitelik-sonuç) sütunlarının ayrılması
X = df.iloc[:,0:-1]
y = df.iloc[:,-1]

# X-y sütunlarının train-test olarak ayrılması
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

# Model sonuçlarının, görselleştirme için kaydedileceği List'ler
model_scores = []
roc_variables = []

##################################################################################
##################################################################################
##################################################################################

# Logistic Regression modelinin oluşturulmasıve eğitilmesi
model_logreg = LogisticRegression(C=10, random_state=RANDOM_SEED)
model_logreg.fit(X_train, y_train)

# Logistic Regression modeli üzerinden X_test verilerinin sonuçlarını tahmin etme
preds_logreg = model_logreg.predict(X_test)

# Logistic Regression modelinin doğruluk (acc) ve hata (MSE) skorlarının alınması
score_logreg = model_logreg.score(X_test, y_test)
mse_logreg = mean_squared_error(y_test, preds_logreg)

# Logistic Regression modelinin sonuçlarının, görselleştirme için List'e kaydedilmesi
model_scores.append(['Logistic Regression', score_logreg, mse_logreg])
roc_variables.append(['Logistic Regression', preds_logreg])

# Logistic Regression modelinin Cross-Validation fonksiyonuna tâbi tutulması
cross_logreg = cross_val_score(model_logreg, X_train, y_train, cv=30)

##################################################################################
##################################################################################
##################################################################################

# Random Forest Classifier modelinin oluşturulmasıve eğitilmesi
model_randomforest = RandomForestClassifier(random_state=RANDOM_SEED)
model_randomforest.fit(X_train, y_train)

# Random Forest Classifier modeli üzerinden X_test verilerinin sonuçlarını tahmin etme
preds_randomforest = model_randomforest.predict(X_test)

# Random Forest Classifier modelinin doğruluk (acc) ve hata (MSE) skorlarının alınması
score_randomforest = model_randomforest.score(X_test, y_test)
mse_randomforest = mean_squared_error(y_test, preds_randomforest)

# Random Forest Classifier modelinin sonuçlarının, görselleştirme için List'e kaydedilmesi
model_scores.append(['Random Forest Classifier', score_randomforest, mse_randomforest])
roc_variables.append(['Random Forest Classifier', preds_randomforest])

# Random Forest Classifier modelinin Cross-Validation fonksiyonuna tâbi tutulması
cross_randomforest = cross_val_score(model_randomforest, X_train, y_train, cv=30)

##################################################################################
##################################################################################
##################################################################################

# Modellerin Confusion Matrix'lerini oluşturma
plot_confusion_matrix(y_test, preds_logreg, 'Logistic Regression')
plot_confusion_matrix(y_test, preds_randomforest, 'Random Forest Tree')

# Modellerin Cross-Validation sonuçlarını görselleştirme
plot_cross_valid('Logistic Regression', cross_logreg, 'yellow')
plot_cross_valid('Random Forest Tree', cross_randomforest, 'red')

# Model sonuçlarının görselleştirilmesi
plot_model_scores(model_scores, colors = ['green', 'yellow'])
plot_ROC(y_test, roc_variables)