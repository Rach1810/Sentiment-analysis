#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

max_words = 200000

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

"""Pre-processing"""

# cut texts after this number of words (among top max_features most common words)
max_review_length = 80

x_train = sequence.pad_sequences(x_train, truncating='pre', padding='pre', maxlen=max_review_length)
x_test = sequence.pad_sequences(x_test, truncating='pre', padding='pre', maxlen=max_review_length)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

"""Build the model"""

print('Build model...')
embedding_length = 64
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=embedding_length, input_length=max_review_length))
model.add(LSTM(units=64, input_shape=(max_review_length, embedding_length), return_sequences=False, unroll=True))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



"""Train the model"""

print('Training...')
batch_size = 32

model.fit(x_train, y_train,
          batch_size = batch_size,
          epochs=10,
          validation_data = (x_test, y_test))

"""Evaluate the model"""

score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print(f'Test score = {score}')
print(f'Test accuracy = {acc}')

"""Making predictions with model"""

#review = "The movie was a great waste of time."
review = "It was a great movie."
print(f'New review = {review}')

d = imdb.get_word_index()
words = review.split()
review = []

for word in words:
  if word not in d:
    review.append(2)
  else:
    review.append(d[word] + 3)
  
print(f"review = {review}")

review = sequence.pad_sequences([review], truncating='pre', padding='pre', maxlen=80)

prediction = model.predict(review)
print(f'Prediction (0 = Negative, 1 = positive) = {prediction}')


# In[107]:


import xlrd
d = imdb.get_word_index()
exc1 = "Movies.xlsx"
b1 = xlrd.open_workbook(exc1)
fs = b1.sheet_by_index(0)
k = [ ]
u = [ ]
r = 0
pred = 0
for i in range(17):
    rv = fs.row_values(i+1)
    print(rv[0])
    w = rv[0].split()
    print(w)
    rv =[ ]
    for word in w:
        if word not in d:
            rv.append(2)
        else:
            rv.append(d[word] + 3)
    rv = sequence.pad_sequences([rv], truncating='pre', padding='pre', maxlen=80)
    print(rv)
    prediction = model.predict(rv)
       
    u.append(prediction[0][0])
    if prediction < 0.5:
        k.append(0)
    else:
        k.append(1)
    print(f'Prediction (0 = Negative, 1 = positive) = {prediction}')
    print(u)


# In[43]:


import pandas as pd
df1 = pd.read_excel("Movies.xlsx")
df1["rating"] = k
df1


# In[46]:


c = 0
a1 = len(k)
print(a1)
for i in k:
    if i==1:
        c +=1
print(c)


# In[48]:


import matplotlib.pyplot as plt

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Positive', 'Negative'
sizes = [c, a1-c]
explode = (0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# In[156]:


import matplotlib.pyplot as plt
import pandas as pd

df1.groupby(['Genre','rating']).size().unstack().plot(kind='bar',stacked=True, color = 'g''r')

plt.show()


# In[108]:


import pandas as pd
df6 = pd.read_excel("Movies.xlsx")
df6["predicted"] = u
df7 = pd.DataFrame(df6)
df6


# In[ ]:





# In[155]:


import matplotlib.pyplot as plt
import pandas as pd

df1.groupby(['Language','rating']).size().unstack().plot(kind='bar',stacked=True, color ='r''b')
plt.title('Language vs. Sentiment')
plt.xlabel('Language')
plt.ylabel('Sentiments')
plt.show()


# In[ ]:




