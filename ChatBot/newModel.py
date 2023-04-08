import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import re 
import numpy
import tflearn
import tensorflow as tf
import random
import json
import pickle
import spacy

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
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

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# tensorflow.reset_default_graph()
tf.compat.v1.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 16, activation="relu")
net = tflearn.fully_connected(net, 32, activation="relu")
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

# Define and train the model
model = tflearn.DNN(net)
model.fit(training, output, n_epoch=4000, batch_size=16, show_metric=True)
model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


products = ["GDP(flower)", "Amnesia Haze(Flower)", "Indica Gummies (Edibles)"]
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

def extract_product_name(user_input):
    product_name = None
    tokens = nltk.word_tokenize(user_input)
    tagged = nltk.pos_tag(tokens)
    for i in range(len(tagged)):
        word, pos = tagged[i]
        if re.match(r'\d+', word) and i < len(tagged) - 1:
            product_name = tagged[i+1][0]
            break
        elif pos.startswith('NN') or pos == 'JJ' and i > 0 and tagged[i-1][1].startswith('NN'):
            product_name = word if not product_name else product_name + ' ' + word
            print(product_name)
    return product_name

def extract_quantity(user_input, pattern):
    match = re.search(pattern, user_input)
    if match:
        return match.group(1)
    return None

def chat():
   
    print("Start talking with the bot (type quit to stop)!")
    while True:
        
        inp = input("You: ")
        model.predict([bag_of_words(inp, words)])
        if inp.lower() == "quit":
            break
        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        print("User input:", inp)
        for tg in data["intents"]:
            print(tag)
            if tg['tag'] == tag:
                responses = tg['responses']
                if tag == "order_product":
                    # define regex pattern to match phrases like "I want to order 2 PRODUCT_NAMEs" and "Can I buy 3 PRODUCT_NAMEs?"
                    patternForQuantity = r"(\d+)"
                    quantity = extract_quantity(inp, patternForQuantity)
                    matchesForProduct = extract_product_name(inp)
                    print(matchesForProduct)
                    print(quantity) 
                    if matchesForProduct is not None:
                        product_name = matchesForProduct
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
                    product_name = extract_product_name(inp)
                    print(product_name, "producr name ")
                    print(get_product_info(product_name))
                    print("you made here ")

                elif tag == "unknown":
                    print("Bot:", random.choice(responses))
                else:
                    print("Bot:", random.choice(responses))
                         
chat()



