import json
#import chardet
with open ("C:\VS CODE\PY projects\intents.json", "r" , encoding = "utf-8") as file:
    data = json.load(file)
    #print(data["intents"])
import json
import string
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
nltk.download('punkt')
nltk.download('wordnet')

words = []
classes = []
data_x = []
data_y = []
for intent in data["intents"]:
  for pattern in intent["patterns"]:
    tokens = nltk.word_tokenize(pattern)
    words.extend(tokens)
    data_x.append(pattern)
    data_y.append(intent["tag"])
  if intent["tag"] not in classes:
    classes.append(intent["tag"])
lemmatizer = WordNetLemmatizer()
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
words = sorted(set(words))
classes = sorted(set(classes))

training = []
out_empty = [0] * len(classes)
for idx , doc in enumerate(data_x):
  bow = []
  text = lemmatizer.lemmatize(doc.lower())
  for word in words:
    bow.append(1) if word in text else bow.append(0)
  output_row = list(out_empty)
  output_row[classes.index(data_y[idx])] = 1
  training.append([bow,output_row])
random.shuffle(training)
training = np.array(training , dtype = object)
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

model = Sequential()
model.add(Dense(128,input_shape = (len(train_x[0]),),activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(64,activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation = "softmax"))
adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-6)
model.compile(loss = 'categorical_crossentropy',optimizer = adam,metrics = ["accuracy"])
print(model.summary())
model.fit(x = train_x,y = train_y,epochs = 150 , verbose = 1)

def clean_text(text):
  tokens = nltk.word_tokenize(text)
  tokens = [lemmatizer.lemmatize(word) for word in tokens]
  return tokens
def bag_of_words(text,vocab):
  tokens = nltk.word_tokenize(text)
  bow = [0]* len(vocab)
  for w in tokens:
    w = lemmatizer.lemmatize(w.lower())
    for idx,word in enumerate(vocab):
      if word == w:
        bow[idx] = 1
  return np.array(bow)
def pred_class(text,vocab,labels):
  bow = bag_of_words(text,vocab)
  result = model.predict(np.array([bow]))[0]
  thresh = 0.5
  y_pred = [[indx,res] for indx,res in enumerate(result) if res > thresh]
  y_pred.sort(key = lambda x: x[1],reverse = True)
  return_list = []
  for r in y_pred :
    return_list.append(labels[r[0]])
  return return_list
def get_response(intents_list , intents_json):
  if len(intents_list) == 0:
    result = "sorry ! I don't understand."
  else:
    tag = intents_list[0]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
      if i["tag"] == tag:
        result = random.choice(i["responses"])
        break
  return result

print("Type quit if you don't wnat to chat with our bot.")
while True:
  message = input("")
  if message == 'quit':
    print("Goodbye , Have a nice dayy ;)")
    break
  intents = pred_class(message,words,classes)
  result = get_response(intents,data)
  print(result)