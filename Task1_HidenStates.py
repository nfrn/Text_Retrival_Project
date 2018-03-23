from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, LSTM, Embedding, RepeatVector, Flatten, Dropout, Activation, TimeDistributed
from nltk.corpus import wordnet as wn
from keras.utils import plot_model

# The max number of words in a question is 30, so the padding into 32 for safety
MAX_WORDS_SENTENCE = 32
MAX_WORDS_ANSWER = 5
WORD_EMBEDDINGS_SIZE = 100


def split_data(sentences):
    answers = sentences[1::2]
    answers_padded = pad_sequences(answers, maxlen=MAX_WORDS_ANSWER, padding='post')

    questions = sentences[::2]
    questions_padded = pad_sequences(questions, maxlen=MAX_WORDS_SENTENCE, padding='post')

    return questions_padded, answers_padded


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

    print(answersT0[0])
    print(answersT1[0])

    embeddings, reverse_word_map = get_embeddings_weights(t, voc_size)

    return embeddings, reverse_word_map, data, answersT1, voc_size, questions, answers, answersT0


def createModel(embeddings, voc_size):
    ### build encoder
    enc_input = Input(shape=(MAX_WORDS_SENTENCE,), dtype='int32', name='encoder_input')
    enc_embed = Embedding(voc_size, WORD_EMBEDDINGS_SIZE, weights=[embeddings], input_length=MAX_WORDS_SENTENCE,
                          trainable=False)(enc_input)
    enc_lstm = LSTM(WORD_EMBEDDINGS_SIZE, return_state=True, return_sequences=True,
                    input_shape=(None, MAX_WORDS_SENTENCE, WORD_EMBEDDINGS_SIZE))
    sequence, state1, state2 = enc_lstm(enc_embed)
    enc_states = [state1, state2]

    ### build decoder

    dec_sequence = LSTM(WORD_EMBEDDINGS_SIZE, return_sequences=True)(sequence, enc_states)

    decoder_dense = Dense(voc_size, activation='softmax')(dec_sequence)

    ### build model
    model = Model(input=enc_input, output=decoder_dense)
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())
    return model


def createModel2(embeddings, voc_size):
    ### build encoder
    enc_input = Input(shape=(MAX_WORDS_SENTENCE,), dtype='int32', name='encoder_input')
    enc_embed = Embedding(voc_size, WORD_EMBEDDINGS_SIZE, weights=[embeddings], input_length=MAX_WORDS_SENTENCE,
                          trainable=False)(enc_input)
    enc_lstm = LSTM(WORD_EMBEDDINGS_SIZE, return_state=True, return_sequences=True,
                    input_shape=(None, MAX_WORDS_SENTENCE, WORD_EMBEDDINGS_SIZE))
    sequence, state1, state2 = enc_lstm(enc_embed)
    enc_states = [state1, state2]

    ### build decoder

    dec_input = Input(shape=(MAX_WORDS_ANSWER,), dtype='int32', name='decoder_input')
    dec_embed = Embedding(voc_size, WORD_EMBEDDINGS_SIZE, weights=[embeddings], input_length=MAX_WORDS_ANSWER,
                          trainable=False)(dec_input)

    dec_sequence = LSTM(WORD_EMBEDDINGS_SIZE, return_sequences=True)(dec_embed, enc_states)

    decoder_dense = Dense(voc_size, activation='softmax')(dec_sequence)

    ### build model
    model = Model(input=[enc_input, dec_input], output=decoder_dense)
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())
    return model


def train(encoder, data, target, voc_size):
    # target = to_categorical(target, num_classes=voc_size)
    target = target.reshape(6795, 32, 1)
    encoder.fit(data, target, epochs=30, batch_size=128, shuffle=False, validation_split=0.2,
                verbose=2)  # validation_data=(data_val, target_val)
    loss, accuracy = encoder.evaluate(data, target, verbose=0)
    print('Accuracy: %f' % (accuracy * 100))
    encoder.save_weights("Q-A_1.h5")

    return encoder


def train2(encoder, data, target, targetT2, voc_size):
    # target = to_categorical(target, num_classes=voc_size)
    target = target.reshape(6795, 5, 1)
    encoder.fit([data, targetT2], target, epochs=30, batch_size=128, shuffle=False, validation_split=0.2,
                verbose=2)  # validation_data=(data_val, target_val)
    loss, accuracy = encoder.evaluate([data, targetT2], target, verbose=0)
    print('Accuracy: %f' % (accuracy * 100))
    encoder.save_weights("Q-A_1.h5")

    return encoder


def testQuestionsToQuestions(data, encoder, reverse_word_map, questions):
    encoder.load_weights("Q-Q_1.h5")
    for i in range(100):
        data2 = data[i].reshape(1, 32)
        sequence = encoder.predict(data2)
        print(sequence.shape)
        print("Question:", end='   ')
        print(questions[i])
        print("Prediction:", end=' ')
        for x in sequence:
            for word in x:
                best = np.argmax(word)
                if best != 0:
                    print(reverse_word_map[best], end=' ')
        print()


def testQuestionsToAnswers(data, encoder, reverse_word_map, answers):
    encoder.load_weights("Q-A_1.h5")
    target_seq = np.zeros((1, MAX_WORDS_ANSWER))
    target_seq[0, 0] = 4
    print(target_seq)
    for i in range(100):
        data2 = data[i].reshape(1, 32)
        sequence = encoder.predict([data2, target_seq])
        print(sequence.shape)
        print("Question:", end='   ')
        print(answers[i])
        print("Prediction:", end=' ')
        for x in sequence:
            for word in x:
                best = np.argmax(word)
                if best != 0:
                    text = reverse_word_map[best]
                    if text != 'end':
                        print(reverse_word_map[best], end=' ')
        print()


def getWupScore():
    sentence = ['one', 'blue', 'chair']
    sentence2 = ['two', 'red', 'table']
    for inx, word in enumerate(sentence):
        synset1 = wn.synsets(word)[0]
        synset2 = wn.synsets(sentence2[inx])[0]
        wups = synset1.wup_similarity(synset2)
        print(wups)


def main():
    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    embeddings, reverse_word_map, data, target, voc_size, questions, answers, answerst1 = load_data("train.txt")

    subtask = 2
    if subtask == 1:
        data = data.reshape(6795, 32)
        model = createModel(embeddings, voc_size)
        plot_model(model, to_file='model.png', show_shapes=True)
        model = train(model, data, data, voc_size)
        testQuestionsToQuestions(data, model, reverse_word_map, questions)

    else:
        data = data.reshape(6795, 32)
        answerst1 = answerst1.reshape(6795, 5)
        model = createModel2(embeddings, voc_size)
        model = train2(model, data, target, answerst1, voc_size)
        testQuestionsToAnswers(data, model, reverse_word_map, answers)


if __name__ == '__main__':
    main()
