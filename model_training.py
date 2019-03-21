import json
import os
import numpy as np
import pickle
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.optimizers import Adam

os.environ["CUDA_DEVICE_ORDER"]= 'PCI_BUS_ID'  
os.environ["CUDA_VISIBLE_DEVICES"]= '1'

RETRAIN = True    
OLD_MODEL_NAME = '2x_gru_model'
NEW_MODEL_NAME = '2x_gru_retrained'
EPOCHS = 20
LR = 0.001

X_train1 = np.load('X_train_120_40_100.npy')
X_train2 = np.load('X_train_120_100_120.npy')
X_train = np.concatenate([X_train1,X_train2])[:5_000_000]
del X_train1
del X_train2

X_test1 = np.load('X_test_120_40_100.npy')
X_test2 = np.load('X_test_120_100_120.npy')
y_train1 = np.load('y_train_120_40_100.npy')
y_train2 = np.load('y_train_120_100_120.npy')
y_test1 = np.load('y_test_120_40_100.npy')
y_test2 = np.load('y_test_120_100_120.npy')



X_test = np.concatenate([X_test1,X_test2])[:1_000_000]
y_train = np.concatenate([y_train1,y_train2])[:5_000_000]
y_test = np.concatenate([y_test1,y_test2])[:1_000_000]

del X_test1
del X_test2
del y_train1
del y_train2
del y_test1
del y_test2

print(X_train.shape, 'X_train')
print(y_train.shape, 'y_train')
print(X_test.shape, 'X_test')
print(y_test.shape, 'y_test')

with open('vocabulary.pkl', 'rb') as f: 
    vocabulary = pickle.load(f)

def clear_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
    K.get_session().graph.get_collection('variables')
    K.clear_session()

def model_load(old_model_name,model_folder):
    clear_session()
    json_file = open(model_folder + old_model_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_folder + old_model_name + '_weights.h5')
    print('model loaded!')
    return model

def datagen(batch_size):
    X_train_copy = np.array([])
    y_train_copy = np.array([])
    while True:
        if X_train_copy.shape[0] == 0:
            X_train_copy = X_train.copy()
            y_train_copy = y_train.copy()
        push_X = X_train_copy[:batch_size]
        push_y = y_train_copy[:batch_size]
        X_train_copy = X_train_copy[batch_size:]
        y_train_copy = y_train_copy[batch_size:]
        yield push_X, push_y
        
def model_construct():
    EMBEDDING_DIM = 2000
    VOCABULARY_SIZE = len(vocabulary)
    MAX_TEXT_LENGTH = 120
    input_length=MAX_TEXT_LENGTH
    text_input = Input((input_length,))
    embed = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBEDDING_DIM, input_length=input_length)(text_input)
    gru_1 = GRU(1024, input_shape=(MAX_TEXT_LENGTH, EMBEDDING_DIM),return_sequences=True)(embed)
    drop_1 = Dropout(0.3)(gru_1)
    gru_2 = GRU(1024,activation='sigmoid')(drop_1)
    drop_2 = Dropout(0.3)(gru_2)
    dense = Dense(VOCABULARY_SIZE, activation='softmax')(drop_2)
    model = Model(text_input, dense)
    print('model construced!')
    return model

def fit(epochs):
    checkpoint = ModelCheckpoint('retrained_fourth-{epoch:02d}-{loss:.4f}.h5', monitor='loss', verbose=1, 
                                 save_best_only=True, mode='min',save_weights_only=True)
    model.fit_generator(datagen(128),steps_per_epoch=5000, 
        epochs=epochs, 
        verbose=1,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint,EarlyStopping()],
    )
    
def save_model(model,model_name):
    model.save_weights(model_name + '_model_weights.h5')
    model_json = model.to_json()
    with open(model_name + '_model.json', "w") as json_file:
        json_file.write(model_json)

        
if RETRAIN:
    model = model_load(OLD_MODEL_NAME,'./')
else:
    model = model_construct()
    
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(LR))

history = fit(EPOCHS)        
save_model(model,NEW_MODEL_NAME)

with open(NEW_MODEL_NAME +'_history.json', 'w') as f:
    json.dump(history, f)



    
