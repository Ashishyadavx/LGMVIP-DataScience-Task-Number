#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[30]:


import tensorflow as tf


# In[31]:


(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()


# In[32]:


plt.figure(figsize=(10,8))
sns.countplot(y_train)


# In[33]:


fig, axes = plt.subplots(ncols=5, sharex=False, 
    sharey=True, figsize=(10, 4))
for i in range(5):
    axes[i].set_title(y_train[i])
    axes[i].imshow(x_train[i], cmap='gray_r')
    axes[i].get_xaxis().set_visible(False)
    axes[i].get_yaxis().set_visible(False)
plt.show()


# In[34]:


print('Train images shape : ',x_train.shape)
print('Test images shape : ',x_test.shape)


# In[35]:


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)


# In[36]:


x_train=x_train/255.0
x_testg=x_test/255.0
num_classes = 10


# In[37]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten,Activation
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from tensorflow.keras.layers import BatchNormalization


# In[38]:


model = Sequential()

model.add(Conv2D(128, kernel_size=(3, 3), input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation(tf.nn.relu))
model.add(Dropout(0.3))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation(tf.nn.relu))
model.add(Dropout(0.3))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(BatchNormalization())
model.add(Activation(tf.nn.relu))
model.add(Dropout(0.3))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation(tf.nn.relu))
model.add(Dropout(0.3))

model.add(Dense(num_classes))
model.add(Activation(tf.nn.softmax))


# In[39]:


model.summary()


# In[40]:


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history=model.fit(x=x_train,y=y_train,validation_split=0.1,epochs=5)


# In[41]:


model.save('Mnist_dataset.h5')


# In[42]:


from tensorflow.keras.models import load_model
model = load_model('Mnist_dataset.h5')


# In[43]:


loss_and_acc=model.evaluate(x_test,y_test)
print("Test Loss", loss_and_acc[0])
print("Test Accuracy", loss_and_acc[1])


# In[45]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])


# In[46]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])


# In[47]:


y_prob=model.predict(x_test)
y_pred=y_prob.argmax(axis=1)


# In[48]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[49]:


y_predicted = model.predict(x_test)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
cm


# In[50]:


plt.imshow(x_test[7],cmap='gray_r')
plt.title('Actual Value: {}'.format(y_test[7]))
prediction=model.predict(x_test)

plt.axis('off')
print('Predicted Value: ',np.argmax(prediction[7]))
if(y_test[7]==(np.argmax(prediction[7]))):
  print('Successful prediction')
else:
  print('Unsuccessful prediction')


# In[51]:



plt.imshow(x_test[1],cmap='gray_r')
plt.title('Actual Value: {}'.format(y_test[1]))
prediction=model.predict(x_test)
plt.axis('off')
print('Predicted Value: ',np.argmax(prediction[1]))
if(y_test[1]==(np.argmax(prediction[1]))):
  print('Successful prediction')
else:
  print('Unsuccessful prediction')

