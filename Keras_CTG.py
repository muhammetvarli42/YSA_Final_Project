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

X[0:1]

X.shape

#%% KERAS YSA


from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense




# Train-Test 
from sklearn.model_selection import train_test_split
# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)








#%% Modeling
# İlkel(Validation yapılmamış) Modeli Kurma
# modeli fonksiyon ile kurmak için fonksiyon
# from keras.optimizers import Adam
# from sklearn.utils import class_weight
#class_weight = class_weight.compute_class_weight('balanced', np.unique(y[:,-1]),y[:,-1])
# Sınıf dengesizliğini gidermek için sınıflara ağırlıklar verildi
class_weight = {0: 1, 1: 5.74, 2: 9.4}



def create_model(optimizer="adam"):
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=23, activation='relu'))
    
    model.add(Dense(40, activation='sigmoid'))
    model.add(Dense(60, activation='relu'))

    
    model.add(Dense(3, activation='softmax')) # 3 output olduğu için output layer 3 olmalı
    # multi-class olduğu için activation fonksiyonu 'softmax' seçilmelidir
    # multi class olduğu için loss fonksiyonu "categorical_crossentropy"
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=["accuracy"])
    return model
model = create_model() # tune edilmemiş model nesnesini oluşturma

egitim=model.fit(X_train, y_train, epochs=100, batch_size=32,class_weight=class_weight, verbose=1,validation_data=(X_test,y_test))




# %% plot loss during training
import matplotlib.pyplot as plt
plt.plot(egitim.history['loss'], label='train')
plt.plot(egitim.history['val_loss'], label='test')
plt.title('Model Loss')
plt.xlabel('epochs')
plt.ylabel('loss values')
plt.legend(loc='upper right')
plt.show()

# %% Modelin Tune Edilmemiş Skorları
import sklearn.metrics as metrics
y_pred=model.predict_classes(X_test)


# %%Accuracy

print("Accuracy:",metrics.accuracy_score(np.argmax(y_test, axis=1),y_pred))

# %%f1 score

print("f1_weighted:",metrics.f1_score(np.argmax(y_test, axis=1), y_pred,average='weighted'))



#%% Grid Search Cross Validation
# GridSearch Cross Validation Parametreleri
param_grid = {
   
    'epochs': [50,100,150], 
    'batch_size':[32,50,100],
    'optimizer':['RMSprop', 'Adam','SGD'],
    
}

# create model
# Model Nesnesini KerasClassifier ile oluşturma işlemi
model_cv = KerasClassifier(build_fn=create_model, verbose=1)


grid = GridSearchCV(estimator=model_cv,  
                    n_jobs=-1, 
                    verbose=1,
                    cv=5,
                    param_grid=param_grid)

grid_cv_model = grid.fit(X_train, y_train,) # GridSearch Nesnesinin Train Seti Üzerinden fit edilmesi


means = grid_cv_model.cv_results_['mean_test_score'] # ortalama test skorları
stds = grid_cv_model.cv_results_['std_test_score'] # test skorlarının standart sapmaları
params = grid_cv_model.cv_results_['params'] # kullanılan parametreler
# oluşan tüm skorları,standart sapmaları ve kullanılan parametreleri ekrana yazdırmak için
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
# Grid Search Cross Validation Sonucunda Ortaya Çıkan En İyi Parametrelerin Ekrana yazdırılması
print("Best: %f using %s" % (grid_cv_model.best_score_, grid_cv_model.best_params_))

#%% Model Tuning- En İyi Parametreler ile Tune Edilmiş Modeli Oluşturma
# Tune Edilmiş Model Nesnesinin KerasClassifier ile oluşturulması
cv_model = grid_cv_model.best_estimator_
 

#%% K-FOLD
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# K-fold accuracy skorları

kfold = KFold(n_splits=5, shuffle=True)
results = cross_val_score(cv_model, X_test, np.argmax(y_test, axis=1), cv=kfold,scoring= 'accuracy')


print('K-fold Cross Validation Accuracy Sonuçları: ', results)
print('K-fold Cross Validation Accuracy Sonuçlarının Ortalaması: ', results.mean())

#%%
# K-fold f1 skorları
from sklearn.model_selection import KFold


kfold = KFold(n_splits=5, shuffle=True)
results = cross_val_score(cv_model, X_test, np.argmax(y_test, axis=1), cv=kfold,scoring="f1_weighted")


print('K-fold Cross Validation f1_weighted Sonuçları: ', results)
print('K-fold Cross Validation f1_weighted Sonuçlarının Ortalaması: ', results.mean())


# %% Tune Edilmiş Model Tahmin 


y_pred = cv_model.predict(X_test) 


# %% f1 score
import sklearn.metrics as metrics
print("f1_weighted:",metrics.f1_score(np.argmax(y_test, axis=1), y_pred,average='weighted'))


# %% Accuracy

print("accuracy:",metrics.accuracy_score(np.argmax(y_test, axis=1), y_pred))
#%% Confusion Matrix and Classification Report
from sklearn.metrics import confusion_matrix, classification_report 

# Classification Report
model_report = classification_report(np.argmax(y_test, axis=1), y_pred)
print(model_report)

# Confusion Matrix
model_conf = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
print(model_conf)



#%% ROC-AUC Curve

y_score = cv_model.predict_proba(X_test)

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