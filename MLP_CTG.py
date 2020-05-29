#%%

from warnings import filterwarnings
filterwarnings('ignore')
import pandas as pd
import numpy as np

df = pd.read_csv("CTG.csv")


df.head()

# Drop unnecessaries
df=df.drop(["FileName","Date","SegFile","b","e","A", "B","C", "D" ,"E", "AD", "DE" ,"LD", "FS", "SUSP"],axis=1)

df.head()

# Coloumns names
df.columns

df.shape

df.isnull().sum()

# tüm  nan verileri silme işlemi
df = df.dropna()

df.isnull().sum()

df.dtypes



## 3 sınıflı model için kullanılacak dataların seçilmesi
X=df.drop(["NSP","CLASS"],axis=1)

y=df["NSP"]

X.head()

nsp_classes = y.unique()
nsp_classes

from keras import utils as np_utils
from sklearn.preprocessing import LabelEncoder
# Encode class values as integers and perform one-hot-encoding
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
y = np_utils.to_categorical(y)
print(y)

y.shape

#%% Veri Standartlaştırma
from sklearn.preprocessing import StandardScaler
Scaler=StandardScaler()
X=Scaler.fit_transform(X)

X[0:3]

X.shape


#%% MLP YSA

from sklearn.model_selection import train_test_split
# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)

from sklearn.neural_network import MLPClassifier # YSA kütüphanesi importu
# İlkel(Validation yapılmamış) Modeli Kurma
mlpc = MLPClassifier(random_state = 0) # YSA model nesnesi oluşturuldu

mlpc.fit(X_train, y_train) # YSA model nesnesi fit edildi


#%% Valide Edilmemiş Model Üzerinden Tahmin İşlemi
y_pred = mlpc.predict(X_test) # model tahmin işlemi test seti üzerinden

import sklearn.metrics as metrics

# %%Accuracy

print("Accuracy:",metrics.accuracy_score(y_test,y_pred))

# %%f1 score

print("f1_weighted:",metrics.f1_score(y_test, y_pred,average='weighted'))

#%% Grid Search Cross Validation

# Cross Validation İşlemi
# CV için parametreler sözlük yapısı şeklinde oluşturuldu
# GİRİLEN PARAMETRELER HAKKINDA BİLGİLER
# alpha : float, default=0.0001 L2 penalty (regularization term) parameter. (ceza parametresi)
 
    
mlpc_params = {"alpha": [0.1, 0.01, 0.0001],
              "hidden_layer_sizes": [(10,10,10),
                                     (100,100,100),
                                     (100,100)],
              "solver" : ["lbfgs","adam","sgd"],
              "activation": ["relu","logistic"]}

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

#%% K-fold f1_weighted

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# K fold
kf = KFold(shuffle=True, n_splits=5) # 5 katlı cv

cv_results_kfold = cross_val_score(mlpc_tuned, X_test, np.argmax(y_test, axis=1), cv=kf, scoring= 'f1_weighted')

print("K-fold Cross Validation f1_weigted Sonuçları: ",cv_results_kfold)
print("K-fold Cross Validation f1_weigted Sonuçlarının Ortalaması: ",cv_results_kfold.mean())

#%% K-fold accuracy

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# K fold
kf = KFold(shuffle=True, n_splits=5) # 5 katlı cv

cv_results_kfold = cross_val_score(mlpc_tuned, X_test, np.argmax(y_test, axis=1), cv=kf, scoring= 'accuracy')

print("K-fold Cross Validation accuracy Sonuçları: ",cv_results_kfold)
print("K-fold Cross Validation accuracy Sonuçlarının Ortalaması: ",cv_results_kfold.mean())


# %% Tune Edilmiş Model Tahmin 
# Final Modelinin test seti üzerinden tahmin işlemi
y_pred = mlpc_tuned.predict(X_test)


# Final Modelinin accuracy değeri
# %% f1 score
import sklearn.metrics as metrics
print("f1_weighted:",metrics.f1_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1),average='weighted'))

# %% Accuracy

print("accuracy:",metrics.accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))

#%% Confusion Matrix and Classification Report
from sklearn.metrics import confusion_matrix, classification_report 

# Classification Report
model_report = classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
print(model_report)

# Confusion Matrix
# multilabel-indicator is not supported bu yüzden np.argmax kullanılmalıdır!
model_conf = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
print(model_conf)




#%% ROC-AUC Curve

y_score = mlpc_tuned.predict_proba(X_test)

from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
# Learn to predict each class against the other


n_classes = 3 # sınıf sayısı




# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

#%% Ozel bir sınıfa ait roc-auc curve çizdirme

plt.figure()
lw = 2 # line_width
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2]) # 2. class değerine göre curve çizdirme
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Eğrisi')
plt.legend(loc="lower right")
plt.show()


#%% Tüm sınıflara ait roc-auc curve çizdirme
from itertools import cycle

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Eğrisinin Çok Sınıfa Genişletilmesi')
plt.legend(loc="lower right")
plt.show()