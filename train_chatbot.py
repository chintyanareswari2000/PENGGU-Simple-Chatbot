import nltk #tool untuk NLP agar bisa tokenize,stemming dkk
from nltk.stem import WordNetLemmatizer # utk melakukan string processing
lemmatizer = WordNetLemmatizer()
import json #import file json berisi chat response bot-user
import pickle #menyimpan data ke sebuah file

import numpy as np #library diperlukan jika ada array (Numerical Python)
from keras.models import Sequential #keras = open source library Deep Learning (menyederhanakan)
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random #mendapatkan random response

words=[]
classes = []  
documents = []
ignore_words = ['?', '!',',','.'] #tidak akan diikutsertakan dalam pengolahan kalimat
data_file = open('intents.json').read()
intents = json.loads(data_file) # masukin data json ke variabel data_file


# proses pengolahan intents di json
for intent in intents['intents']:
    for pattern in intent['patterns']:

        #tokenize setiap kata dalam kalimat
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #menambahkan data ke corpus = set of texts
        documents.append((w, intent['tag']))

        # menambahkan ke classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lakukan lemmatization , ubah ke huruf kecil dan hapus duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))


# sort classes
classes = sorted(list(set(classes)))


# documents = combination antara patterns dan intents
print (len(documents), "documents")
# classes = intents
print (len(classes), "classes", classes)
# words = all words, vocabulary
print (len(words), "unique lemmatized words", words)


pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# training data
training = []
# array untuk output nanti
output_empty = [0] * len(classes)


# training set
for doc in documents:
    # initialize bagofwords
    bag = []
    # list kata-kata yg sdh di tokenize 
    pattern_words = doc[0]
    # lemmatize setiap kata yang ada dan mengubah menjadi kata dasar
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # membuat bagofwords, jika match maka array diisi 1
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # output nol untuk setiap tag dan 1 untuk current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])



random.shuffle(training)
training = np.array(training)


# intents membuat test lists
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")



model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#menyimpan model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created")

