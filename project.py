from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras import Sequential
from keras.layers import Dense, LSTM, Embedding, Flatten
import tensorflow as tf



#inspired in https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

#The max number of words in a question is 30, so the padding into 32 for safety
MAX_WORDS_SENTENCE=32

def split_data(sentences,voc_size):

    data = np.zeros((6795, MAX_WORDS_SENTENCE))
    target = np.zeros((6795, MAX_WORDS_SENTENCE))
    counter1 = 0
    counter2 = 0
    for i in range(len(sentences)):
        if i % 2 == 0:
            data[counter1]= sentences[i]
            counter1+=1
        else:
            target[counter2]= sentences[i]
            counter2+=1

    return data, target

def get_embeddings_weights(t, vocab_size):

    embeddings_index = dict()
    f = open('glove.6B.100d.txt', encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))

    # create a weight matrix for words in training docs
    embedding_matrix = np.zeros((vocab_size, 100))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def load_data(path):

    with open(path, 'r') as document:
        sentences = (document.read().splitlines())

    t = Tokenizer()
    t.fit_on_texts(sentences)

    encoded_sentences = t.texts_to_sequences(sentences)
    padded_sentences = pad_sequences(encoded_sentences, maxlen=MAX_WORDS_SENTENCE, padding='post')    #[13590*100]

    voc_size = len(t.word_index) + 1
    data, target = split_data(padded_sentences,voc_size)             #[6795*100]  #[6795*100]

    embeddings = get_embeddings_weights(t, voc_size)

    return embeddings, data, target, voc_size



def createModel(embeddings, voc_size):
    model = Sequential()
    e = Embedding(voc_size, 100, weights=[embeddings], input_length=MAX_WORDS_SENTENCE, trainable=False)
    model.add(e)
    model.add(LSTM(100, input_shape=(MAX_WORDS_SENTENCE,100), return_sequences=True))
    model.add(Flatten())
    model.add(Dense(MAX_WORDS_SENTENCE, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def trainModel(model, data, target):  #, data_val, target_val

    model.fit(data,target, nb_epoch=1000, batch_size=100, verbose=2, )  #validation_data=(data_val, target_val)
    loss, accuracy = model.evaluate(data, target, verbose=0)
    print('Accuracy: %f' % (accuracy * 100))
def main():

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    embeddings, data, target, voc_size = load_data("train.txt")
    model = createModel(embeddings, voc_size)
    trainModel( model, data, target)
    model.save_weights("my_model.h5")



if __name__ == '__main__':
    main()
