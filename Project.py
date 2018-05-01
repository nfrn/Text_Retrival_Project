from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, LSTM, Embedding
from nltk.corpus import wordnet as wn
import datetime
import csv

#Which subtask(1 or 2) to run and with or without training(0 or 1)
SUBTASK = 1
TRAIN = 0

#Sizes need to represent the Questions and Answers
MAX_WORDS_QUESTION = 32
MAX_WORDS_ANSWER = 8
WORD_EMBEDDINGS_SIZE = 100

#Evaluation metrics, to avoid the padding bias
TESTS_FOR_ACCURACY = 100
WUP_THRESHOLD = 0.9

#Training parameters
EPOCHS = 50
BATCH_SIZE = 128
VALIDATION = 0.2

#Image Features
IMG_FEA_PATH = "img_features.csv"
IMG_TOTAL = 1449
IMG_FEA_TOTAL = 2048

#Model files
if TRAIN == 1:
    Q_Q_MODEL = "Q-Q_" + str(datetime.date.today()) + ".h5"
    Q_A_MODEL = "Q-A_" + str(datetime.date.today()) + ".h5"
else:
    Q_Q_MODEL = "Q-Q.h5"
    Q_A_MODEL = "Q-A.h5"

TRAIN_DATA = "train.txt"
EMBEDDINGS_DATA = "glove.6B.100d.txt"

#Model parameters
ACTIVATION = 'softmax'
LOSS_FUNCTION = 'sparse_categorical_crossentropy'
OPTIMIZER = 'rmsprop'


def split_data(sentences):
    answers = sentences[1::2]
    answers_padded = pad_sequences(answers, maxlen=MAX_WORDS_ANSWER, padding='post')

    questions = sentences[::2]
    questions_padded = pad_sequences(questions, maxlen=MAX_WORDS_QUESTION, padding='post')

    return questions_padded, answers_padded


def get_embeddings_weights(t, vocab_size):
    embeddings_index = dict()
    f = open(EMBEDDINGS_DATA, encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))

    # create a weight matrix for words in training docs
    embedding_matrix = np.zeros((vocab_size, WORD_EMBEDDINGS_SIZE))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    reverse_word_map = dict(map(reversed, t.word_index.items()))
    return embedding_matrix, reverse_word_map


def load_data(path):
    with open(path, 'r') as document:
        doc = document.read().split('\n')

    doc = doc[:-1]  # ignore last blank sentence

    # fil with ini and end tokens
    sentences = []
    questions = []
    answers = []
    answersT1 = []
    answersT0 = []
    for idx, line in enumerate(doc):

        if idx % 2 == 1:
            answersT1.append(line + ' End')
            answersT0.append('Start ' + line)
            answers.append(line)
            line = 'Start ' + line + ' End'
            sentences.append(line)
        else:
            sentences.append(line)
            questions.append(line)

    t = Tokenizer()
    t.fit_on_texts(sentences)

    encoded_sentences = t.texts_to_sequences(sentences)

    voc_size = len(t.word_index) + 1
    data, target = split_data(encoded_sentences)

    answersT1 = t.texts_to_sequences(answersT1)
    answersT0 = t.texts_to_sequences(answersT0)

    answersT0 = pad_sequences(answersT0, maxlen=MAX_WORDS_ANSWER, padding='post')
    answersT1 = pad_sequences(answersT1, maxlen=MAX_WORDS_ANSWER, padding='post')

    embeddings, reverse_word_map = get_embeddings_weights(t, voc_size)

    return embeddings, reverse_word_map, data, answersT1, voc_size, questions, answers, answersT0


def createModel(embeddings, voc_size):
    ### build encoder
    enc_input = Input(shape=(MAX_WORDS_QUESTION,), dtype='int32', name='encoder_input')
    enc_embed = Embedding(voc_size, WORD_EMBEDDINGS_SIZE, weights=[embeddings], input_length=MAX_WORDS_QUESTION,
                          trainable=False)(enc_input)
    enc_lstm = LSTM(WORD_EMBEDDINGS_SIZE, return_state=True, return_sequences=True,
                    input_shape=(None, MAX_WORDS_QUESTION, WORD_EMBEDDINGS_SIZE))
    sequence, state1, state2 = enc_lstm(enc_embed)
    enc_states = [state1, state2]

    ### build decoder

    dec_sequence = LSTM(WORD_EMBEDDINGS_SIZE, return_sequences=True)(sequence, enc_states)

    decoder_dense = Dense(voc_size, activation=ACTIVATION)(dec_sequence)

    ### build model
    model = Model(input=enc_input, output=decoder_dense)
    model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=['accuracy'])

    print(model.summary())
    return model


def createModel2(embeddings, voc_size):
    ### build encoder
    enc_input = Input(shape=(MAX_WORDS_QUESTION,), dtype='int32', name='encoder_input')
    enc_embed = Embedding(voc_size, WORD_EMBEDDINGS_SIZE, weights=[embeddings], input_length=MAX_WORDS_QUESTION,
                          trainable=False)(enc_input)
    enc_lstm = LSTM(WORD_EMBEDDINGS_SIZE, return_state=True, return_sequences=True,
                    input_shape=(None, MAX_WORDS_QUESTION, WORD_EMBEDDINGS_SIZE))
    sequence, state1, state2 = enc_lstm(enc_embed)
    enc_states = [state1, state2]

    ### build decoder

    dec_input = Input(shape=(MAX_WORDS_ANSWER,), dtype='int32', name='decoder_input')
    dec_embed = Embedding(voc_size, WORD_EMBEDDINGS_SIZE, weights=[embeddings], input_length=MAX_WORDS_ANSWER,
                          trainable=False)(dec_input)

    dec_sequence = LSTM(WORD_EMBEDDINGS_SIZE, return_sequences=True)(dec_embed, enc_states)

    decoder_dense = Dense(voc_size, activation=ACTIVATION)(dec_sequence)

    ### build model
    model = Model(input=[enc_input, dec_input], output=decoder_dense)
    model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=['accuracy'])

    print(model.summary())
    return model


def train(encoder, data, target):
    target = target.reshape(target.shape[0], MAX_WORDS_QUESTION, 1)
    encoder.fit(data, target, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=False, validation_split=VALIDATION,
                verbose=2)
    encoder.save_weights(Q_Q_MODEL)

    return encoder


def train2(encoder, data, target, target2):
    target = target.reshape(target.shape[0], MAX_WORDS_ANSWER, 1)
    encoder.fit([data, target2], target, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=False, validation_split=VALIDATION,
                verbose=2)

    encoder.save_weights(Q_A_MODEL)

    return encoder


def testQuestionsToQuestions(data, encoder, reverse_word_map, questions):
    encoder.load_weights(Q_Q_MODEL)
    for i in range(TESTS_FOR_ACCURACY):
        data2 = data[i].reshape(1, MAX_WORDS_QUESTION)
        sequence = encoder.predict(data2)
        print("Question:", end='   ')
        print(questions[i])
        print("Prediction:", end=' ')
        for x in sequence:
            for word in x:
                best = np.argmax(word)
                if best != 0:
                    print(reverse_word_map[best], end=' ')
        print()
    target = data.reshape(data.shape[0], MAX_WORDS_QUESTION, 1)
    loss, accuracy = encoder.evaluate(data, target, verbose=0)
    print('Accuracy: %f' % (accuracy * 100))


def testQuestionsToAnswers(data, encoder, reverse_word_map, answers,answerst1):
    encoder.load_weights(Q_A_MODEL)
    target_seq = np.zeros((1, MAX_WORDS_ANSWER))
    target_seq[0, 0] = 4
    predicted_values = []
    for i in range(TESTS_FOR_ACCURACY):
        data2 = data[i].reshape(1, MAX_WORDS_QUESTION)
        sequence = encoder.predict([data2, target_seq])
        sequence_to_compare = []
        print("Answer:  ", end='   ')
        print(answers[i])
        print("Prediction:", end=' ')
        for x in sequence:
            for word in x:
                best = np.argmax(word)
                sequence_to_compare.append(best)
                if best != 0:
                    text = reverse_word_map[best]
                    if text != 'end':
                        print(reverse_word_map[best], end=' ')
        print()
        predicted_values.append(sequence_to_compare)

    showRealAccuracyScore(predicted_values[:TESTS_FOR_ACCURACY],answerst1[:TESTS_FOR_ACCURACY])
    showWupsScore(predicted_values[:TESTS_FOR_ACCURACY],answerst1[:TESTS_FOR_ACCURACY],reverse_word_map)


def showRealAccuracyScore(predictions,original):
    correct_counter=0
    for x in range(TESTS_FOR_ACCURACY):
        equal_values = np.sum(predictions[x] == original[x])
        if(equal_values==MAX_WORDS_ANSWER):
            correct_counter+=1
    acc = ((correct_counter/TESTS_FOR_ACCURACY)*100)
    print("Accuracy for 100 examples from the training data: ", end=' ')
    print("%.2f " % acc)

def showWupsScore(predictions,original,mapping):
    finalWups = 0
    for x in range(TESTS_FOR_ACCURACY):
        wupValue = 0
        counterWords = 0
        for word in range(MAX_WORDS_ANSWER):
            wordA = predictions[x][word]
            wordB = original[x][word]
            if wordA!=0:
                if wordB!=0:
                    wupValue += getWupScore(mapping[wordA],mapping[wordB])
                else:
                    wupValue += 0
                counterWords += 1
        finalWups += (wupValue / counterWords)
    finalWups = ((finalWups/TESTS_FOR_ACCURACY)*100)
    print("Wups for 100 examples from the training data: ", end=' ')
    print("%.2f " % finalWups)


def getWupScore(wordA, wordB):
    #inspired from https://datasets.d2.mpi-inf.mpg.de/mateusz14visual-turing/calculate_wups.py

    global_weight_A = 1
    global_weight_B = 1

    global_weight = min(global_weight_A, global_weight_B)

    if wordA=='' or wordB=='':
        return 0

    if wordA==wordB:
        return global_weight * 1

    local_weight_A = 1
    local_weight_B = 1

    semantic_field_A = wn.synsets(wordA, pos=wn.NOUN)
    semantic_field_B = wn.synsets(wordB, pos=wn.NOUN)

    if semantic_field_A==[] or semantic_field_B==[]:
        return 0

    global_max=0.0
    for a in semantic_field_A:
        for b in semantic_field_B:
            local_score = a.wup_similarity(b)
            if local_score > global_max:
                global_max=local_score

    if global_max < WUP_THRESHOLD:
        interp_weight = 0.1
    else:
        interp_weight = 1.0

    final_score = global_max*local_weight_A*local_weight_B*interp_weight*global_weight
    return final_score

def main():
    image_features = loadImageFeatures()

    embeddings, reverse_word_map, data, target, voc_size, questions, answers, answerst1 = load_data(TRAIN_DATA)

    if SUBTASK == 1:
        data = data.reshape(data.shape[0], MAX_WORDS_QUESTION)

        model = createModel(embeddings, voc_size)
        if (TRAIN):
            model = train(model, data, data)
        testQuestionsToQuestions(data, model, reverse_word_map, questions)

    elif SUBTASK == 2:
        data = data.reshape(data.shape[0], MAX_WORDS_QUESTION)
        answerst1 = answerst1.reshape(answerst1.shape[0], MAX_WORDS_ANSWER)

        model = createModel2(embeddings, voc_size)
        if(TRAIN):
            model = train2(model, data, target, answerst1)
        testQuestionsToAnswers(data, model, reverse_word_map, answers, target)

def loadImageFeatures():
    image_features = np.zeros((IMG_TOTAL,IMG_FEA_TOTAL))
    with open(IMG_FEA_PATH, 'r', encoding="utf8") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for idx, row in enumerate(spamreader):
            image_features[idx] = row[1::]

    return image_features

if __name__ == '__main__':
    main()