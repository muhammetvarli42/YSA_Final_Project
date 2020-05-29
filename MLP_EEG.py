
from warnings import filterwarnings
filterwarnings('ignore')
import pandas as pd


df=pd.read_csv("eeg_clean.csv")

df.head()

df.info()

df.isnull().sum()

print(df["eye"].value_counts())

df.eye=[1 if each =="Open" else 0 for each in df.eye]

df.info()

y = df["eye"].values
X = df.drop(['eye'], axis=1).values

#%% Veri Standartlaştırma
from sklearn.preprocessing import StandardScaler
Scaler=StandardScaler()
X=Scaler.fit_transform(X)

X[0:3]
#%% MLP YSA

from sklearn.model_selection import train_test_split
# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0)



from sklearn.neural_network import MLPClassifier # YSA kütüphanesi importu

mlpc = MLPClassifier(random_state = 0) # YSA model nesnesi oluşturuldu

mlpc.fit(X_train, y_train) # YSA model nesnesi fit edildi
#%% Valide Edilmemiş Model Üzerinden Tahmin İşlemi
y_pred = mlpc.predict(X_test) # model tahmin işlemi test seti üzerinden

import sklearn.metrics as metrics
# %%Accuracy

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# %%f1 score

print("f1:",metrics.f1_score(y_test, y_pred))


#%% Grid Search Cross Validation

# Cross Validation İşlemi
# CV için parametreler sözlük yapısı şeklinde oluşturuldu
# GİRİLEN PARAMETRELER HAKKINDA BİLGİLER
# alpha : float, default=0.0001 L2 penalty (regularization term) parameter. (ceza parametresi)

    
mlpc_params = {"alpha": [0.1, 0.01, 0.001],
              "hidden_layer_sizes": [(100,100),
                                     (100,100,100)],
              "solver" : ["adam","sgd"],
              "activation": ["relu","logistic"]
              }
from sklearn.model_selection import GridSearchCV




mlpc = MLPClassifier(random_state = 0) # Model nesnesi oluşturuldu
# Model CV etme 
mlpc_cv_model = GridSearchCV(mlpc, mlpc_params, 
                         cv = 5, # 5 Katlı CV yapılması için
                         n_jobs = -1, # işlemciyi tam performansta kullanıma olanak sağlar
                         verbose = 2) # işlemciyi tam performansta kullanıma olanak sağlar

mlpc_cv_model.fit(X_train, y_train) # Model fit etme işlemi 10 sn sürdü
# fit ederken scale edilmiş train seti üzerinden fit ettik!!

# CV sonucunda elde edilen en iyi parametre
print("En iyi parametreler: " + str(mlpc_cv_model.best_params_))

#%% Model Tuning
# Final Modelinin en iyi parametre ile kurulması
mlpc_tuned = mlpc_cv_model.best_estimator_
# Final Modelinin fit edilmesi
mlpc_tuned.fit(X_train, y_train)


#%% K-fold accuracy

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# K fold
kf = KFold(shuffle=True, n_splits=5) # 5 katlı cv

cv_results_kfold = cross_val_score(mlpc_tuned, X_test, y_test, cv=kf)

print("K-fold Cross Validation Accuracy Sonuçları: ",cv_results_kfold)
print("K-fold Cross Validation Accuracy Sonuçlarının Ortalaması: ",cv_results_kfold.mean())
#%% K fold f1
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


kf = KFold(shuffle=True, n_splits=5) # 5 katlı cv

cv_results_kfold = cross_val_score(mlpc_tuned, X_test, y_test, cv=kf, scoring= 'f1')

print("K-fold Cross Validation f1 Sonuçları: ",cv_results_kfold)
print("K-fold Cross Validation f1 Sonuçlarının Ortalaması: ",cv_results_kfold.mean())


# %% Tune Edilmiş Model Tahmin 
# Final Modelinin test seti üzerinden tahmin işlemi
y_pred = mlpc_tuned.predict(X_test)



# %% f1 score
import sklearn.metrics as metrics
print("f1:",metrics.f1_score(y_test,y_pred))

# %% Accuracy

print("accuracy:",metrics.accuracy_score(y_test,y_pred))
#%% Confusion Matrix and Classification Report
from sklearn.metrics import confusion_matrix, classification_report 

# Classification Report
model_report = classification_report(y_test, y_pred)
print(model_report)

# Confusion Matrix
model_conf = confusion_matrix(y_test, y_pred)
print(model_conf)



#%% ROC-AUC Curve
import matplotlib.pyplot as plt



probs=mlpc_tuned.predict_proba(X_test)
fpr,tpr,threshold=metrics.roc_curve(y_test,y_pred)
roc_auc=metrics.auc(fpr,tpr)




plt.title("ROC")
plt.plot(fpr,tpr,label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy',  linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()