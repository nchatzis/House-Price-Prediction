#!/usr/bin/env python
# coding: utf-8

# # Πρόβλεψη τιμών ακινήτων
# 

# <font size="4"> Load τις βιβλιοθήκες

# In[22]:


#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.layers import Dense, Activation
from keras.models import Sequential
import numpy as np

print('Setup complete')


# <font size="4"> Αρχικά φορτώνουμε τα δεδομένα από το csv αρχείο σε ένα pandas dataframe και με τη συνάρτηση head() βλέπουμε τις πρώτες 5 γραμμές

# In[23]:


df = pd.read_csv('housing.csv')
df.head()


# <font size='4'> Η συνάρτηση describe μας δίνει μερικές στατιστικές πληροφορίες για τα δεδομένα

# In[24]:


df.describe()


# Ενώ η συνάρτηση info() μας λέει το είδος δεδομένων κάθε στήλης

# In[25]:


df.info()


# <font size='4'>Ξεχωρίζουμε τα χαρακτηρηστικά στις μεταβλητές X και y. Στην μεταβλητή y θα μπει το χαρακτηρηστικό στόχος 'median_house_value' 

# In[26]:


X = df[['longitude','latitude','housing_median_age','total_rooms','total_bedrooms',
        'population','households','median_income','ocean_proximity']]
y = df[['median_house_value']]


# <font size='4'>Ξεχωρίζουμε τα αριθμητικά χαρακτηρηστικά από τα κατηγορικά

# In[27]:


categorical = [col for col in X.columns if X[col].dtype=='object']
numerical = [col for col in X.columns if X[col].dtype!='object']


# ## Οπτικοποίηση δεδομένων
# 
# ### Ιστογράμματα
# 
# <font size='4'>Χρησιμοποιούμε τη συνάρτηση hist() που δημιουργεί ιστόγραμμα για κάθε στήλη του dataframe

# In[28]:


df.hist(bins=50, figsize=(20,15))
plt.show()


# ### Δισδιάστατα γραφήματα 
# 
# <font size='4'>Χρησιμοποιούμε τη συνάρτηση plot με παράμετρο kind="scatter". Δοκιμάζουμε κάθε χαρακτηρηστικό του Χ με το χαρακτηρηστικό "median_house_value"

# In[29]:


df.plot(kind="scatter",x="longitude",y="median_house_value")
df.plot(kind="scatter",x="latitude",y="median_house_value")
df.plot(kind="scatter",x="housing_median_age",y="median_house_value")
df.plot(kind="scatter",x="total_rooms",y="median_house_value")
df.plot(kind="scatter",x="total_bedrooms",y="median_house_value")
df.plot(kind="scatter",x="population",y="median_house_value")
df.plot(kind="scatter",x="households",y="median_house_value")
df.plot(kind="scatter",x="median_income",y="median_house_value")
plt.show()


# ## Προεπεξεργασία δεδομένων
# <br>
# 
# <font size='4'>
#     <strong>1) Ελλιπείς τιμές</strong>
# 
# Χρησιμοποιούμε τη συνάρτηση isnull() για να μετρήσουμε τις τιμές που λείπουν σε κάθε στήλη

# In[30]:


missing = df.isnull().sum()
print(missing)
missing.plot.bar()


# <font size='4'>Χρησιμοποιούμε τον SimpleImputer με strategy="median" για να γεμίσουμε τις ελλιπείς τιμές με τη διάμεση τιμή και μετά αντικαθιστούμε τις παλιές στήλες του Χ (που έχουν nan τιμές) με τις νέες (που δεν έχουν)

# In[31]:


imputer = SimpleImputer(strategy="median")
X_imp = pd.DataFrame(imputer.fit_transform(X[numerical]),columns=numerical)

X_temp = X.drop(numerical,axis=1)
X = pd.concat([X_temp,X_imp],axis=1)

print(X.isnull().sum())


# <font size='4'>και τώρα γέμισαν οι θέσεις με τις κενές τιμές</font>
# 
# ### 2) Κλιμάκωση
# 
# <font size='4'>Χρησιμοποιούμε την τεχνική MinMax για να βρίσκονται όλα τα δεδομένα μας στην κλίμακα 0-1
#     https://en.wikipedia.org/wiki/Feature_scaling

# In[32]:


scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X[numerical]),columns=numerical)

X_temp = X.drop(numerical,axis=1)
X = pd.concat([X_temp,X_scaled],axis=1)

X.head()


# <font size='4'>Επαναλαμβάνουμε για το y

# In[33]:


y = pd.DataFrame(scaler.fit_transform(y),columns=y.columns)
y.head()


# <font size='4'> Αν κάνουμε describe() τώρα θα δούμε πως οι μέγιστες και οι ελάχιστες τιμές είναι παντού 0 και 1 αντίστοιχα.

# In[34]:


X.describe()


# In[35]:


y.describe()


# ### 3) One-Hot-Encoding
# 
# Χρησιμοποιούμε την τεχνική One-Hot-Encoding για να αντιμετωπίσουμε το κατηγορικό χαρακτηρηστικό 'ocean_proximity'

# In[36]:


oc_prox = X['ocean_proximity'].unique()
encoder = OneHotEncoder(handle_unknown='ignore',sparse=False)    #sparse=False?
X_enc = pd.DataFrame(encoder.fit_transform(X[categorical]),columns=oc_prox)
#X_enc.index = X.index

X_temp = X.drop(categorical,axis=1)
X = pd.concat([X_temp,X_enc],axis=1)
X.head()


# ## Παλινδρόμηση δεδομένων
# 
# ### 1) Αλγόριθμος Ελάχιστου Μέσου Τετραγωνικού Σφάλματος (Least Mean Squares)
# 
# <font size='4'> Υλοποιήσαμε τις συναρτήσεις lms_train και lms_predict για training του θ και πρόβλεψη αντίστοιχα

# In[37]:


def lms_train(X,y,r=0.1):
    theta_curr = np.zeros((X.shape[1],1))
    theta_prev = np.ones((X.shape[1],1))
    counter=0
    n = len(y)

    while (counter<100):#απότομη κατ΄άβαση
        theta_prev = theta_curr
        y_pred = np.dot(X,theta_prev)
        theta_curr = theta_prev - (r/n)*X.T.dot(y_pred-y)
        counter += 1
    return theta_curr


def lms_predict(X,theta):
    return np.matmul(X,theta)


# <font size='4'>Χρησιμοποιούμε τη συνάρτηση KFold για την 10-πλη διεπικύρωση και τις συναρτήσεις mean_squared_error και mean_absolute_error για το μέσο τετραγωνικό σφάλμα και το μέσο απόλυτο σφάλμα αντίστοιχα (σελ 116 βιβλίου)

# In[38]:


kf = KFold(n_splits=10)
kf.get_n_splits(X)
k=1


for train_index,test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    theta = lms_train(X_train.to_numpy(),y_train.to_numpy())
    pred = lms_predict(X_test.to_numpy(),theta)

    print(f'MSE for Fold Number: {k}, {mean_squared_error(y_test.to_numpy(), pred)}')
    print(f'MAE for Fold Number: {k}, {mean_absolute_error(y_test.to_numpy(), pred)}')
    print("\n")
    k=k+1


# ### 2) Αλγόριθμος Ελάχιστου Τετραγωνικού Σφάλματος (Least Squares)
# 
# <font size='4'> Υλοποιήσαμε τις συναρτήσεις least_squares_train και least_squares_predict για την εύρεση του θ και την πρόβλεψη αντίστοιχα (σελ 118 βιβλίου)

# In[39]:


def least_squares_train(X,y):
    mul1 = X.T.dot(X)
    inv1 = np.linalg.pinv(mul1)
    mul2 = X.T.dot(y)
    theta = np.matmul(inv1,mul2)
    return theta

def least_squares_predict(X,w):
    return np.matmul(X,w)


# <font size='4'> Σε κάθε iteration του loop κάνουμε train στο training set και predict στο test set

# In[40]:


kf = KFold(n_splits=10)
kf.get_n_splits(X)
k=1


for train_index,test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    w = least_squares_train(X_train.to_numpy(),y_train.to_numpy())
    pred2 = least_squares_predict(X_test.to_numpy(),w)

    print(f'MSE for Fold Number: {k}, {mean_squared_error(y_test.to_numpy(), pred2)}')
    print(f'MAE for Fold Number: {k}, {mean_absolute_error(y_test.to_numpy(), pred2)}')
    print("\n")

    k=k+1


# ### 3) Νευρωνικό δίκτυο
# 
# <font size='4'> Υλοποιήσαμε ένα νευρωνικό δίκτυο με τη βιβλιοθήκη keras. Το ΝΔ έχει ένα Input layer, 2 hidden layers μεγέθους 13 και ένα output layer. Χρησιμοποιούμε μόνο 5 epochs για λόγους ταχύτητας (τα σφάλματα είναι μικρά) 

# In[41]:


k=1

for train_index,test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = Sequential()
    model.add(Dense(13, activation = 'relu', input_dim = 13))
    model.add(Dense(units = 13, activation = 'relu'))#hidden
    model.add(Dense(units = 13, activation = 'relu'))#hidden
    model.add(Dense(units = 1))

    model.compile(optimizer = 'adam',loss = 'mean_squared_error')

    model.fit(X_train, y_train, batch_size = 10, epochs = 5)

    y_pred = model.predict(X_test)

    print(f'MSE for Fold Number: {k}, {mean_squared_error(y_pred,y_test)}')
    print(f'MAE for Fold Number: {k}, {mean_absolute_error(y_pred,y_test)}')
    print("\n")

    k=k+1

