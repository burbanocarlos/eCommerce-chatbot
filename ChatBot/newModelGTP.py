#deep learning library
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.preprocessing.text import Tokenizer

from keras.utils import pad_sequences
import pandas as pd
import numpy as np
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import string
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import re

#read CSV file
train = pd.read_csv('/Users/carlosburbano/Documents/Projects/ChatBot/Train.csv')
valid = pd.read_csv('/Users/carlosburbano/Documents/Projects/ChatBot/Valid.csv')

#read intents file
with open('intents.json') as file:
    data = json.load(file)

#combine all sentences from both files
labels = []
all_sentences = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        all_sentences.append(pattern)

for intent in data["intents"]:
    labels.append(intent["tag"])
for sentence in train['text']:
    all_sentences.append(sentence)
for sentence in valid['text']:
    all_sentences.append(sentence)

#Tokenize the sentences
tokenizer = Tokenizer(num_words=120000)

words = tokenizer.word_index.keys()
 #preparing vocabulary
tokenizer.fit_on_texts(all_sentences)
#converting text into integer sequences
train_seq = tokenizer.texts_to_sequences(train['text'])
valid_seq = tokenizer.texts_to_sequences(valid['text'])
#padding to prepare sequences of same length
train_seq = pad_sequences(train_seq, maxlen=100)
valid_seq = pad_sequences(valid_seq, maxlen=100)

size_of_vocabulary=len(tokenizer.word_index) + 1
try:
    # Load the saved model
    model = load_model('best_model.h5')
    print('Loaded saved model')
except:
    # define the model
    model = Sequential()

    # embedding layer
    model.add(Embedding(len(tokenizer.word_index)+1, 300, input_length=max_length, trainable=True)) 

    # convolutional layer with 32 filters, kernel size of 3, and ReLU activation
    model.add(Conv1D(32, 3, activation='relu'))
    # max pooling layer
    model.add(MaxPooling1D())

    # convolutional layer with 64 filters, kernel size of 3, and ReLU activation
    model.add(Conv1D(64, 3, activation='relu'))
    # max pooling layer
    model.add(MaxPooling1D())

    # flatten layer
    model.add(Flatten())

    # dense layer with 128 neurons and ReLU activation
    model.add(Dense(128, activation='relu'))

    # output layer with softmax activation
    model.add(Dense(num_classes, activation='softmax'))

    # compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["acc"])

    #Print summary of model
print(model.summary())


# def bag_of_words(sentence, tokenizer):
#     # remove punctuation marks from sentence
#     sentence = sentence.translate(str.maketrans('', '', string.punctuation))
#     # tokenize the sentence
#     sentence_words = nltk.word_tokenize(sentence.lower())
#     # initialize the bag of words
#     bag = [0] * 100
#     # loop through each word in the sentence and check if it's in the words list
#     for i, s in enumerate(sentence_words):
#         if i >= 100:
#             break
#         if s in tokenizer.word_index:
#             bag[i] = tokenizer.word_index[s]
#     return np.array(bag)
def bag_of_words(sentence, words, max_length=100):
    # tokenize the sentence
    tokens = nltk.word_tokenize(sentence)
    # stem each word
    stemmed_tokens = [stemmer.stem(word.lower()) for word in tokens]
    # initialize bag with zeros
    bag = np.zeros(max_length, dtype=np.int32)
    # loop through words in sentence
    for idx, w in enumerate(words):
        if w in stemmed_tokens:
            # set the corresponding index in bag to 1 if word is found
            bag[idx] = 1
    return bag


    
   







products = ["Grand Daddy Purple(Flower)", "Amnesia Haze(Flower)", "Indica Gummies (Edibles)"]
dict_products = {"grand daddy purple" : 1234,
                 "amnesia haze" : 3456,
                 "indica gummies" : 7897
                }
products_lower = ["grand daddy purple", "amnesia haze", "indica gummies"]
product = [   
     {     
        "name": "grand daddy purple",   
        "price": 10.99,      
        "description": "This is a description for Product A."    }, 
     {      
        "name": "amnesia haze",   
        "price": 19.99,     
        "description": "This is a description for Product B."   },   
     {      
        "name": "indica gummies",        
        "price": 7.99,        
        "description": "This is a description for Product C."   
     }]
orderNumbers = []


def get_order_status(order_number):
    for order in orderNumbers:
        if order == order_number:
            return "Your order is being processed."
    else:
        return "Sorry, we could not find your order."

def get_product_info(product_name):
    for item in product:
        if item['name'] == product_name:
            return f"{item['name']}: {item['description']}, Price: {item['price']}"

    return "Sorry, that product is not available."

def order_product(product_name, quantity=1):
    product_name_lower = product_name.lower()
    if product_name_lower in products_lower:
        product_index = products_lower.index(product_name_lower)
        product_price = product[product_index]['price']
        total_price = product_price * int(quantity)
        print(f"Bot: Do you confirm the sale of {quantity} {product_name}(s) for a total of {total_price}?")
        confirm = input("You: ")
        if confirm.lower() in ["yes", "y"]:
            order_number = random.randint(100000, 999999)
            orderNumbers.append(order_number)
            print(f"Bot: Your order has been confirmed. Order number: {order_number}")
        else:
            print("Bot: Okay, your order has been cancelled.")
    else:
        print("Bot: Sorry, that product is not available.")

def extract_product_name(text):
    product_name = None
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    for i in range(len(tagged)):
        word, pos = tagged[i]
        if pos.startswith('NN') or pos == 'JJ' and i > 0 and tagged[i-1][1].startswith('NN'):
            product_name = word if not product_name else product_name + ' ' + word
    return product_name

def chat():
   
    print("Start talking with the bot (type quit to stop)!")
    while True:
        max_length = 100
        inp = input("You: ")
        input_array = np.zeros((1, max_length), dtype=np.int32)
        bow = bag_of_words(inp, words)
        
        input_array[0, :len(bow)] = bow
        print(input_array, "input")
        results = model.predict([input_array])
        results_index = np.argmax(results)
    


        # Pad the input text to a fixed length of 100 tokens
        # inp = " ".join(inp[:100]) + ' <PAD>' * (100 - len(inp[:100]))
        # input_array = np.array([tokenizer.enconde_plus(inp, add_special_tokens=True)])
        # print(input_string)
        print(len(tokenizer.word_index))

        # print(input_array.shape)



        
        if inp.lower() == "quit":
            break
        
        tag = labels[results_index]
        print("User input:", inp)
        for tg in data["intents"]:
            print(tag)
            if tg['tag'] == tag:
                responses = tg['responses']
                if tag == "order_product":
                    # define regex pattern to match phrases like "I want to order 2 PRODUCT_NAMEs" and "Can I buy 3 PRODUCT_NAMEs?"
                    pattern = re.compile(r"(?i)(?:i\s+want\s+to\s+order|can\s+i\s+buy|can\s+you\s+help\s+me\s+place\s+an\s+order\s+for|how\s+can\s+i\s+order|what's\s+the\s+next\s+step\s+to\s+order|can\s+you\s+guide\s+me\s+through\s+the\s+order\s+process\s+for)\s+(?P<quantity>\d+)\s+(?P<product_name>.+?)s?\??$")
                    match = pattern.search(inp)
                    if match:
                        product_name = match.group("product_name")
                        quantity = match.group("quantity")
                        if quantity:
                            order_product(product_name, quantity.strip())
                        else:
                            order_product(product_name)
                    else:
                        # handle case where product name is not found
                        print("Sorry, I didn't catch the product name. Can you please try again?")
                elif tag == 'order_status':
                    print(random.choice(responses))
                    order_number = input("You: ")
                    print("Bot:", get_order_status(int(order_number)))
                elif tag == "product_info":
                    # check if any of the patterns match
                   # pattern to match product_name
                    pattern = r"\{product_name\}"

                    # find all matches of pattern in user input
                    matches = extract_product_name(inp)

                    # extract product_name from matches
                    print(matches)

                    # print product_name
                    #print(product_name)
                    print("you made here ")

                elif tag == "unknown":
                    print("Bot:", random.choice(responses))
                else:
                    print("Bot:", random.choice(responses))
                         
chat()

