from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
from keras import Sequential
from keras.layers import Dense, LSTM, Embedding, RepeatVector, Flatten, Dropout, Activation,TimeDistributed
import tensorflow as tf
from nltk.corpus import wordnet as wn



#inspired in https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

#The max number of words in a question is 30, so the padding into 32 for safety
MAX_WORDS_SENTENCE=32
WORD_EMBEDDINGS_SIZE=100

def split_data(sentences):
    print(len(sentences))
    data = np.zeros((6795, MAX_WORDS_SENTENCE))
    target = np.zeros((6795, MAX_WORDS_SENTENCE))
    counter1 = 0
    counter2 = 0
    for i in range(len(sentences)):
        if i % 2 == 1:
            target[counter2] = sentences[i]
            counter2 += 1
        else:
            data[counter1] = sentences[i]
            counter1 += 1

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


    reverse_word_map = dict(map(reversed, t.word_index.items()))
    return embedding_matrix, reverse_word_map


def load_data(path):

    with open(path, 'r') as document:
        doc = document.read().split('\n')

    doc=doc[:-1] # ignore last blank sentence

    #fil with ini and end tokens
    sentences=[]
    questions=[]
    for idx, line in enumerate(doc):

        if idx % 2 == 1:
            answer = 'START ' + line + ' END'
            sentences.append(answer)
        else:
            sentences.append(line)
            questions.append(line)

    t = Tokenizer()
    t.fit_on_texts(sentences)

    encoded_sentences = t.texts_to_sequences(sentences)

    padded_sentences = pad_sequences(encoded_sentences, maxlen=MAX_WORDS_SENTENCE, padding='post')    #[13590*100]

    voc_size = len(t.word_index) + 1
    data, target = split_data(padded_sentences)             #[6795*100]  #[6795*100]

    embeddings, reverse_word_map = get_embeddings_weights(t, voc_size)

    return embeddings, reverse_word_map, data, target, voc_size, questions



def createModel(embeddings, voc_size):
    model = Sequential()

    model.add(Embedding(voc_size, WORD_EMBEDDINGS_SIZE, weights=[embeddings], trainable=False))

    model.add(LSTM(WORD_EMBEDDINGS_SIZE, input_shape=(MAX_WORDS_SENTENCE, WORD_EMBEDDINGS_SIZE) , return_sequences=True))

    model.add(Dense(voc_size, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print(model.summary())
    return model


def train(encoder,data,target,voc_size):
    #target = to_categorical(target, num_classes=voc_size)
    target = target.reshape(6795, 32, 1)
    encoder.fit(data, target, epochs=30, batch_size=128, shuffle=False, validation_split=0.2,
                verbose=2)  # validation_data=(data_val, target_val)
    loss, accuracy = encoder.evaluate(data, target, verbose=0)
    print('Accuracy: %f' % (accuracy * 100))
    encoder.save_weights("withEmbeddings.h5")

    return encoder

def testQuestionsToQuestions(data,encoder,reverse_word_map,questions):

    encoder.load_weights("withEmbeddings.h5")
    for i in range(100):
        data2 = data[i].reshape(1, 32)
        sequence = encoder.predict_classes(data2)
        print(sequence.shape)
        print("Question:",end='   ')
        print(questions[i])
        print("Prediction:",end=' ')
        for x in sequence:
            for word in x:
                if word!=0:
                    print(reverse_word_map[word],end=' ')
        print()

def getWupScore():
    sentence = ['one', 'blue', 'chair']
    sentence2 = ['two', 'red', 'table']
    for inx,word in enumerate(sentence):
        synset1 = wn.synsets(word)[0]
        synset2 = wn.synsets(sentence2[inx])[0]
        wups = synset1.wup_similarity(synset2)
        print(wups)

def main():

    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    embeddings, reverse_word_map, data, target, voc_size, questions = load_data("train.txt")

    data = data.reshape(6795, 32)
    encoder = createModel(embeddings, voc_size)

    #encoder = train(encoder,data,data,voc_size)

    testQuestionsToQuestions(data,encoder,reverse_word_map,questions)



if __name__ == '__main__':
    main()
