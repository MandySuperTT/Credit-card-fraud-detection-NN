# CREDIT CARD FRAUD DETECTION
# STEP #1: IMPORTING DATA
# import libraries 
import pandas as pd
import numpy as np
import keras

np.random.seed(2)
data = pd.read_csv('creditcard.csv')

# STEP #2: VISUALIZATION OF THE DATA
# data.head()

# STEP #3: PRE-PROCESSING
from sklearn.preprocessing import StandardScaler
data['normalizedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data = data.drop(['Amount'],axis=1)

data = data.drop(['Time'],axis=1)

X = data.iloc[:, data.columns != 'Class']
y = data.iloc[:, data.columns == 'Class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=0)

min_train = X_train.min()
range_train = (X_train - min_train).max()
X_train_scaled = (X_train - min_train)/range_train

min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test

# STEP #4: TRAINING THE MODEL
from keras.models import Sequential
from keras.layers import Dense,LeakyReLU
from keras.layers import Dropout

model = Sequential()
model.add(Dense(input_dim = 29,
                output_dim = 16))
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.25))
model.add(Dense(25))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(15))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(1,
               activation='sigmoid'))
# model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=128,epochs=5)

# STEP #5: EVALUATING THE MODEL
score = model.evaluate(X_test, y_test)
print(score)
# [0.0037883689044154864, 0.9993328885923949]

predicted_classes = model.predict_classes(X_test)
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_test, predicted_classes)
sns.heatmap(cm,annot=True)
cnf_matrix = confusion_matrix(y_test, predicted_classes.round())
print(cnf_matrix)
# [[85281    15]
# [   42   105]]
