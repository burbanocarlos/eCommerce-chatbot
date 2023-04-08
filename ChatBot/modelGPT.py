import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel



nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()


def bag_of_words(tokenized_words, words):
    tokenized_words = [stem(word) for word in tokenized_words]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in tokenized_words:
            bag[idx] = 1.0
    return bag

def tokenize_words(words):
    return nltk.word_tokenize(words)

def stem(word):
    return lemmatizer.lemmatize(word.lower())
with open("intents.json") as file:
    intents = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:




    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            tokenized_words = tokenize_words(pattern)
            words.extend(tokenized_words)
            docs_x.append(tokenized_words)
            docs_y.append(intent['tag'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    words = sorted(set([stem(word) for word in words]))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = bag_of_words(doc, words)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# ... rest of the code

try:
    model = load_model('model.h5')

except:
    # Create and fit the model
    tf.compat.v1.reset_default_graph()

    model = Sequential()
    model.add(Dense(128, input_shape=(len(training[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(output[0]), activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(training, output, epochs=200, batch_size=5, verbose=1)

    model.save('model.h5')

    with open('label_encoder.pickle', 'wb') as le_pickle:
        pickle.dump(LabelEncoder().fit(labels), le_pickle)


print(model.summary())



def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict(np.array([bag_of_words(inp, words)]))
        results_index = np.argmax(results)
        tag = labels[results_index]

        for tg in intents["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        print(random.choice(responses))


if __name__ == '__main__':
    chat()