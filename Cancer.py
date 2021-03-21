# Mengimpor library
import pandas as pd
import seaborn as sns
sns.set()
 
# Mengimpor dataset
from sklearn.datasets import load_breast_cancer
kanker = load_breast_cancer()
 
# Menggabungkan ke dalam satu data frame
df_kanker = pd.DataFrame(kanker['data'], columns = kanker['feature_names'])
df_kanker['status'] = kanker['target']
 
# Visualisasi data
sns.pairplot(df_kanker, vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'], hue = 'status' )
 
# Plot korelasi dengan heatmap
sns.heatmap(df_kanker[['mean radius', 'mean texture', 'concavity error', 'mean area','mean smoothness']].corr(), annot=True, cmap='bwr')
 
# Persiapan training dataset
X = df_kanker.drop(['status'],axis=1)
y = df_kanker['status']
 
# Membagi dataset ke training set dan test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
 
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
 
# Proses training dengan SVM
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
clf = SVC(kernel='rbf', random_state=0)
clf.fit(X_train, y_train)
 
# Mengevauasi model
y_predict = clf.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot=True)
print(classification_report(y_test,y_predict))