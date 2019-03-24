import telebot
import random
import pickle
import os
import json
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.models import model_from_json

os.environ["CUDA_DEVICE_ORDER"]= 'PCI_BUS_ID'  
os.environ["CUDA_VISIBLE_DEVICES"]= '1'

json_file = open('private_data.json', 'r')
private_data = json.loads(json_file.read())
bot = telebot.TeleBot(private_data['api_token'])

UNKNOWN_INDEX = 1
SEQUENCE_START = '<START>'
SEQUENCE_END = '<END>'
MAX_TEXT_LENGTH = 120


with open('vocabulary.pkl', 'rb') as f: 
    vocabulary = pickle.load(f)

token_by_index = {index: token for token, index in vocabulary.items()}

SEQUENCE_START_INDEX = vocabulary[SEQUENCE_START]
SEQUENCE_END_INDEX = vocabulary[SEQUENCE_END]

json_file.close()


def clear_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
    K.get_session().graph.get_collection('variables')
    K.clear_session()
    
def model_load():
    clear_session()
    json_file = open('2x_gru_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights('2x_gru_model_weights.h5')
    print('model loaded!')
    return model    

def generate():
    j = 0
    indices = [vocabulary[SEQUENCE_START]]
    while True:
        sequence = pad_sequences([indices], MAX_TEXT_LENGTH)

        predictions = model.predict(sequence)[0]
        j += 1
        if (predictions.argmax() == SEQUENCE_END_INDEX):
            words_list = ['.']
            dot_flag = True
            for index in indices[1:]:
                word = token_by_index[index]
                if word == '.':
                    dot_flag = True
                elif dot_flag:
                    word = word.capitalize()
                    dot_flag = False
                    
                if word in [')','.',',']:
                    words_list[-1] = words_list[-1] + word
                elif words_list[-1] == '(':
                    words_list[-1] = words_list[-1] + word
                else:
                    words_list.append(word)
            return ' '.join(words_list[1:])
        
        seed = random.random()
        
        total = 0
        for i, probability in enumerate(predictions):
            total += probability
            if seed < total and i not in [UNKNOWN_INDEX, SEQUENCE_START_INDEX, SEQUENCE_END_INDEX] and i != indices[-1]:
                indices.append(i)
                break
                

recipe_str = 'recipe 🍩🍰'
info_str =  'info 🔞'
more_str = 'more, please 🙏'
author_str = 'author 😎'

info_but = telebot.types.KeyboardButton(info_str)
author_but = telebot.types.KeyboardButton(author_str)
more_but = telebot.types.KeyboardButton(more_str)
recipe_but = telebot.types.KeyboardButton(recipe_str)

start_markup = telebot.types.ReplyKeyboardMarkup(row_width=1)
start_markup.add(recipe_but, more_but)

main_markup = telebot.types.ReplyKeyboardMarkup(row_width=3)
main_markup.add(info_but,author_but,more_but, recipe_but)

@bot.message_handler(commands=['start'])
def send_welcome(message):
    welcome_message = "Hey! I'm a recipe bot v 0.4. and I'm trying to generate desserts and baking recipes🍪🍧 😁\n"
    bot.send_message(message.chat.id, welcome_message, reply_markup=start_markup)
    
@bot.message_handler(func=lambda message: message.text.lower() == more_str)
def send_main(message):
    main_message = 'info - how does it work👷\nauthor - creator contacts😎'
    bot.send_message(message.chat.id, main_message, reply_markup=main_markup)

@bot.message_handler(func=lambda message: message.text.lower() == info_str)
def send_info(message):
    info_message = "For those who know - I am a simple recurrent neural network using 2 gru layers with embedding input vectors.\n"
    info_message += 'Embedding dim is 120, activations - tanh, sigmoid🛀\n'
    info_message += 'For others - I am a neural network, who is trying to remember combinations of words from human-made recipes.🍲\n '
    info_message += 'I have no idea how to create a new word, but I guarantee that each recipe is unique😍'
    bot.send_message(message.chat.id, info_message)
    
@bot.message_handler(func=lambda message: message.text.lower() == author_str)
def send_author(message):
    author_message = " telegram @masmx86\n vk https://vk.com/id179091229"
    bot.send_message(message.chat.id, author_message, reply_markup=main_markup)
    
    
    
@bot.message_handler(func=lambda message: message.text.lower() in  ['recipe','рецепт','generate',recipe_str])
def send_generate(message):
    with graph.as_default():
        bot.send_message(message.chat.id, 'please, wait...⏳')
        bot.send_message(message.chat.id, generate(), )
        bot.send_message(message.chat.id, 'gotcha⬆⬆\n🍻🍻',reply_markup=main_markup)
        
@bot.message_handler(func=lambda message: True)
def send_undefined(message):
    send_welcome(message)
    bot.send_message(message.chat.id, 'I have no idea what do you want 😡\n Please use buttons')

model = model_load()
global graph
graph = tf.get_default_graph()




bot.infinity_polling(True) #uncomment this and comment cycle below if you have a stable network

#while True:
#    try:
        #bot.polling(True)
#    except:
#        continue
