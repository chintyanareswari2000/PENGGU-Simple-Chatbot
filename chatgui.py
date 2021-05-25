import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np



from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):
    # lakukan tokenization - split kata yang ada lalu masukkan ke array
    sentence_words = nltk.word_tokenize(sentence)
    # lakukan stemming 
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array dengan 0 atau 1 jika ada kata yang terdapat di bag itu ditemukan di kalimatnya

def bow(sentence, words, show_details=True):
    # tokenize patternya
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                

                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))



def predict_class(sentence, model):
    
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    


    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


# pembuatan GUI dengan menggunakan tkinter

import tkinter
from tkinter import *


def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "Kamu : " + msg + '\n\n')
        ChatLog.config(foreground="#df7861", font=("Courier", 12 ))
    
        res = chatbot_response(msg)
        ChatLog.insert(END, "PENGGU : " + res + '\n\n')
            
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
 

base = Tk()
base.title("PENGGU")
base.geometry("500x500")
base.resizable(width=FALSE, height=FALSE)





#buat label atas
l = Label(base, text="Selamat Datang di PENGGU!")
l.config(font= 'Courier 12 bold')
k = Label(base, text="email: petchinggu@gmail.com | phone: +62837463453")
k.config(font= 'Courier 10')

# buat chat window
ChatLog = Text(base, bd=0, bg="#fcf8e8", height="8", width="300", font="Courier",)

ChatLog.config(state=DISABLED)

# buat scrollbar di samping kanan chat window
scrollbar = Scrollbar(base, command=ChatLog.yview)
ChatLog['yscrollcommand'] = scrollbar.set

#buat button send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#ecdfc8", activebackground="#ecb390",fg="#df7861",
                    command= send )

# buat box input chat dari user
EntryBox = Text(base, bd=0, bg="#fcf8e8",width="29", height="5", font="Courier",fg = "#df7861")
#EntryBox.bind("<Return>", send)


# penempatan komponen windows
l.pack()
k.pack()
scrollbar.place(x=480,y=6, height=386)
ChatLog.place(x=6,y=44, height=348, width=480)
EntryBox.place(x=6, y=401, height=90, width=370)
SendButton.place(x=355, y=401, height=90)



base.mainloop()
