import pickle


with open('classes.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data)

with open('words.pkl', 'rb') as e:
    data = pickle.load(e)
    print(data)