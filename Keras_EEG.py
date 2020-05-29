
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

#%% KERAS YSA


from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense,Dropout
from keras import Sequential


from sklearn.model_selection import train_test_split
# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0)


#%% Modeling
# İlkel(Validation yapılmamış) Modeli Kurma

# modeli fonksiyon ile kurmak için fonksiyon
def create_model(optimizer="adam"):
    # create model
    model = Sequential()
    model.add(Dense(32, input_dim=14, activation='relu'))
    model.add(Dropout(0.1))# overfittingin önüne geçmek için droupout eklendi
    model.add(Dense(20, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='sigmoid'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid')) # 2 outputlu olduğu için çıktı katmanında aktivasyon olarak 'sigmoid'
   
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=["accuracy"])
    return model


model = create_model() # tune edilmemiş model nesnesini oluşturma

egitim=model.fit(X_train, y_train, epochs=100, batch_size=50, verbose=1,validation_data=(X_test,y_test))

# %% plot loss during training
import matplotlib.pyplot as plt
plt.plot(egitim.history['loss'], label='train')
plt.plot(egitim.history['val_loss'], label='test')
plt.title('model loss')
plt.xlabel('epochs')
plt.ylabel('loss values')
plt.legend(loc='upper right')
plt.show()

# %% Modelin Tune Edilmemiş Skorları
import sklearn.metrics as metrics
y_pred=model.predict_classes(X_test)
# %%Accuracy

print("Accuracy:",metrics.accuracy_score(y_test,y_pred))

# %%f1 score

print("f1:",metrics.f1_score(y_test, y_pred))


#%% Grid Search Cross Validation

# GridSearch Cross Validation Parametreleri
param_grid = {
   
    'epochs': [100,150],
    'batch_size':[32,50],
    'optimizer':['Adam','SGD'],
    
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
results = cross_val_score(cv_model, X_test, y_test, cv=kfold,scoring= 'accuracy')


print('K-fold Cross Validation Accuracy Sonuçları: ', results)
print('K-fold Cross Validation Accuracy Sonuçlarının Ortalaması: ', results.mean())

#%%
# K-fold f1 skorları
from sklearn.model_selection import KFold


kfold = KFold(n_splits=5, shuffle=True)
results = cross_val_score(cv_model, X_test, y_test, cv=kfold,scoring="f1")


print('K-fold Cross Validation f1 Sonuçları: ', results)
print('K-fold Cross Validation f1 Sonuçlarının Ortalaması: ', results.mean())

# %% Tune Edilmiş Model Tahmin (Grid Search CV ile oluşan model)
y_pred= cv_model.predict(X_test)



# %% f1 score
import sklearn.metrics as metrics
print("f1:",metrics.f1_score(y_test, y_pred))

# %% Accuracy

print("acc:",metrics.accuracy_score(y_test, y_pred))
#%% Confusion Matrix and Classification Report
from sklearn.metrics import confusion_matrix, classification_report 

# Classification Report
model_report = classification_report(y_test, y_pred)
print(model_report)

# Confusion Matrix
model_conf = confusion_matrix(y_test, y_pred)
print(model_conf)

#%% ROC-AUC

probs=cv_model.predict_proba(X_test)

fpr,tpr,threshold=metrics.roc_curve(y_test,y_pred)
roc_auc=metrics.auc(fpr,tpr)


plt.title("ROC")
plt.plot(fpr,tpr,label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy',  linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()