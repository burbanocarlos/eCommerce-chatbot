import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import re
import numpy as np
import string

nltk.download('averaged_perceptron_tagger')
with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickleV3", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)
    
    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickleV3", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 16)
net = tflearn.fully_connected(net, 16)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=2000, batch_size=16, show_metric=True)
model.save("model.tflearnV3")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

products = ["Grand Daddy Purple(Flower)", "Amnesia Haze(Flower)", "Indica Gummies (Edibles)"]

dict_products = {"GDP" : 1234,
                 "amnesia haze" : 3456,
                 "indica gummies" : 7897
                }
products_lower = ["gdp", "amnesia haze", "indica gummies"]
product = [   
     {     
        "name": "gdp",   
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


def extract_product_name(text):
    # Assuming product name can consist of multiple words
    words = text.split()

    for word in words:
        if word in products_lower:
            print(word)
            return word
        else:
            return None

# Helper function to extract product name and quantity from user input
def extract_product_name_and_quantity(text):
    pattern = re.compile(r'(?P<quantity>\d+)\s+(?P<product_name>(\w+\s*)+)')
    match = pattern.search(text)
    
    if match:
        quantity = int(match.group('quantity'))
        product_name = match.group('product_name').strip()
        print(product_name, quantity)
        return product_name, quantity
    else:
        return None, None
    

# Improved order_product function
def order_product(product_name, quantity=1):
    product_name_lower = product_name.lower()
    if product_name_lower in products_lower:
        product_index = products_lower.index(product_name_lower)
        product_price = product[product_index]['price']
        total_price = product_price * int(quantity)
        print(f"Bot: Do you confirm the sale of {quantity} {product_name}(s) for a total of {total_price}?")
        confirm = input("You: ")
    else:
        print("Bot: Sorry, that product is not available.")

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

# Improved chat function
def chat():
    print("Start talking with the bot (type quit to stop)!")
    confidence_threshold = 0.75
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        confidence = results[0][results_index]
        if confidence < confidence_threshold:
            print("Bot: I'm not sure what you mean. Can you please rephrase that?")
            continue
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
                if tag == "order_product":
                    product_name, quantity = extract_product_name_and_quantity(inp)
                    if product_name and quantity:
                        order_product(product_name, quantity)
                    elif product_name:
                        order_product(product_name)
                    else:
                        print("Sorry, I didn't catch the product name. Can you please try again?")
                elif tag == 'order_status':
                    print(random.choice(responses))
                    order_number = input("You: ")
                    print("Bot:", get_order_status(int(order_number)))
                elif tag == "product_info":
                    product_name = extract_product_name(inp)
                    if product_name:
                        print("Bot:", get_product_info(product_name.lower()))
                    else:
                        print("Sorry, I didn't catch the product name. Can you please try again?")
                elif tag == "unknown":
                    print("Bot:", random.choice(responses))
                else:
                    print("Bot:", random.choice(responses))


chat()
