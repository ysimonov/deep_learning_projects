import json 
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import colorama
from colorama import Fore, Style
import random
import pickle
import os 

colorama.init()

currend_dir = os.path.dirname(os.path.abspath(__file__))

intents_path = os.path.normpath(os.path.join(currend_dir, 'intents.json'))
with open(intents_path, 'r') as file:
    data = json.load(file)

def chat():

    # load trained model
    chat_path = os.path.normpath(os.path.join(currend_dir, 'chat_model'))
    model = keras.models.load_model(chat_path)

    # load tokenizer
    tonenizer_path = os.path.normpath(os.path.join(currend_dir, 'tokenizer.pickle'))
    with open(tonenizer_path, 'rb') as file:
        tokenizer = pickle.load(file)

    # load label encoder
    encoder_path = os.path.normpath(os.path.join(currend_dir, 'label_encoder.pickle'))
    with open(encoder_path, 'rb') as file:
        label_encoder = pickle.load(file)

    # parameters
    max_len = 20

    while True:
        print(Fore.LIGHTBLACK_EX + "User: ", Style.RESET_ALL, end="")
        input_message = input()
        if input_message.lower() in ('q', "quit", "exit"):
            break
        result = model.predict(
            keras.preprocessing.sequence.pad_sequences(
                tokenizer.texts_to_sequences([input_message]),
                truncating='post', maxlen=max_len
            ))
        tag = label_encoder.inverse_transform([np.argmax(result)])
        for i in data['intents']:
            if i['tag'] == tag:
                print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL , np.random.choice(i['responses']))
                break

print(Fore.YELLOW + "Start messaging with the bot (type quit to stop)!" + Style.RESET_ALL)
chat()