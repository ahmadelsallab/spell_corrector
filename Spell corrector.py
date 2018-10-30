
# coding: utf-8

# # Introduction
# 
# We tackle the problem of OCR post processing. In OCR, we map the image form of the document into the text domain. This is done first using an CNN+LSTM+CTC model, in our case based on tesseract. Since this output maps only image to text, we need something on top to validate and correct language semantics.
# 
# The idea is to build a language model, that takes the OCRed text and corrects it based on language knowledge. The langauge model could be:
# - Char level: the aim is to capture the word morphology. In which case it's like a spelling correction system.
# - Word level: the aim is to capture the sentence semnatics. But such systems suffer from the OOV problem.
# - Fusion: to capture semantics and morphology language rules. The output has to be at char level, to avoid the OOV. However, the input can be char, word or both.
# 
# The fusion model target is to learn:
# 
#     p(char | char_context, word_context)
# 
# In this workbook we use seq2seq vanilla Keras implementation, adapted from the lstm_seq2seq example on Eng-Fra translation task. The adaptation involves:
# 
# - Adapt to spelling correction, on char level
# - Pre-train on a noisy, medical sentences
# - Fine tune a residual, to correct the mistakes of tesseract 
# - Limit the input and output sequence lengths
# - Enusre teacher forcing auto regressive model in the decoder
# - Limit the padding per batch (TODO)
# - Learning rate schedule (TODO)
# 


from __future__ import print_function
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras import optimizers
from keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np
import os
from sklearn.model_selection import train_test_split


# # Utility functions


# Limit gpu allocation. allow_growth, or gpu_fraction
def gpu_alloc():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))



gpu_alloc()



def calculate_WER_sent(gt, pred):
    '''
    calculate_WER('calculating wer between two sentences', 'calculate wer between two sentences')
    '''
    gt_words = gt.lower().split(' ')
    pred_words = pred.lower().split(' ')
    d = np.zeros(((len(gt_words) + 1), (len(pred_words) + 1)), dtype=np.uint8)
    # d = d.reshape((len(gt_words)+1, len(pred_words)+1))

    # Initializing error matrix
    for i in range(len(gt_words) + 1):
        for j in range(len(pred_words) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(gt_words) + 1):
        for j in range(1, len(pred_words) + 1):
            if gt_words[i - 1] == pred_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    return d[len(gt_words)][len(pred_words)]



def calculate_WER(gt, pred):
    '''

    :param gt: list of sentences of the ground truth
    :param pred: list of sentences of the predictions
    both lists must have the same length
    :return: accumulated WER
    '''
#    assert len(gt) == len(pred)
    WER = 0
    nb_w = 0
    for i in range(len(gt)):
        #print(gt[i])
        #print(pred[i])
        WER += calculate_WER_sent(gt[i], pred[i])
        nb_w += len(gt[i])

    return WER / nb_w

def noise_maker(sentence, threshold):
    '''Relocate, remove, or add characters to create spelling mistakes'''
    letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m',
           'n','o','p','q','r','s','t','u','v','w','x','y','z',]
    noisy_sentence = []
    i = 0
    while i < len(sentence):
        random = np.random.uniform(0, 1, 1)
        # Most characters will be correct since the threshold value is high
        if random < threshold:
            noisy_sentence.append(sentence[i])
        else:
            new_random = np.random.uniform(0, 1, 1)
            # ~33% chance characters will swap locations
            if new_random > 0.67:
                if i == (len(sentence) - 1):
                    # If last character in sentence, it will not be typed
                    continue
                else:
                    # if any other character, swap order with following character
                    noisy_sentence.append(sentence[i + 1])
                    noisy_sentence.append(sentence[i])
                    i += 1
            # ~33% chance an extra lower case letter will be added to the sentence
            elif new_random < 0.33:
                random_letter = np.random.choice(letters, 1)[0]
                noisy_sentence.append(random_letter)
                noisy_sentence.append(sentence[i])
            # ~33% chance a character will not be typed
            else:
                pass
        i += 1

    return ''.join(noisy_sentence)


def load_data_with_gt(file_name, num_samples, max_sent_len, min_sent_len):
    '''Load data from txt file, with each line has: <TXT><TAB><GT>. The  target to the decoder muxt have \t as the start trigger and \n as the stop trigger.'''
    cnt = 0  
    input_texts = []
    gt_texts = []
    target_texts = []
    for row in open(file_name):
        if cnt < num_samples :
            sents = row.split("\t")
            input_text = sents[0]
            
            target_text = '\t' + sents[1] + '\n'
            if len(input_text) > min_sent_len and len(input_text) < max_sent_len and len(target_text) > min_sent_len and len(target_text) < max_sent_len:
                cnt += 1
                
                input_texts.append(input_text)
                target_texts.append(target_text)
                gt.append(sents[1])
    return input_texts, target_texts, gt_texts



def load_data_with_noise(file_name, num_samples, noise_threshold, max_sent_len, min_sent_len):
    '''Load data from txt file, with each line has: <TXT>. The GT is just a noisy version of TXT. The  target to the decoder muxt have \t as the start trigger and \n as the stop trigger.'''
    cnt = 0  
    input_texts = []
    gt_texts = []
    target_texts = []
    for row in open(file_name):
        if cnt < num_samples :
            input_text = noise_maker(row, noise_threshold)
            input_text = input_text[:-1]
            gt.append(row)
            target_text = '\t' + row + '\n'            
            if len(input_text) > min_sent_len and len(input_text) < max_sent_len and len(target_text) > min_sent_len and len(target_text) < max_sent_len:
                cnt += 1
                input_texts.append(input_text)
                target_texts.append(target_text)
    return input_texts, target_texts, gt_texts


def build_vocab(all_texts):
    '''Build vocab dictionary to victorize chars into ints'''
    vocab_to_int = {}
    count = 0
    
    for sentence in all_texts:
        for char in sentence:
            if char not in vocab_to_int:
                vocab_to_int[char] = count
                count += 1
    # Add special tokens to vocab_to_int
    codes = ['\t','\n']
    for code in codes:
        if code not in vocab_to_int:
            vocab_to_int[code] = count
            count += 1
    '''''Build inverse translation from int to char'''
    int_to_vocab = {}
    for character, value in vocab_to_int.items():
        int_to_vocab[value] = character
        
    return vocab_to_int, int_to_vocab


def vectorize_data(input_texts, target_texts, max_encoder_seq_length, num_encoder_tokens, vocab_to_int):
    '''Prepares the input text and targets into the proper seq2seq numpy arrays'''
    encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
    decoder_input_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')
    decoder_target_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            # c0..cn
            encoder_input_data[i, t, vocab_to_int[char]] = 1.
        for t, char in enumerate(target_text):
            # c0'..cm'
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, vocab_to_int[char]] = 1.
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, vocab_to_int[char]] = 1.
                
    return encoder_input_data, decoder_input_data, decoder_target_data

def decode_sequence(input_seq, encoder_model, decoder_model, num_decoder_tokens, int_to_vocab):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, vocab_to_int['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = int_to_vocab[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence
def decode_texts(input_texts, model, latent_dim, vocab_to_int, max_encoder_seq_length, num_encoder_tokens):
    # Vicotrize data
    encoder_input_data, decoder_input_data, decoder_target_data = vectorize_data(input_texts=input_texts,
                                                                                 target_texts=target_texts, 
                                                                                 max_encoder_seq_length=max_encoder_seq_length, 
                                                                                 num_encoder_tokens=num_encoder_tokens, 
                                                                                 vocab_to_int=vocab_to_int)
    # Model
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_encoder_tokens)) 
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)   

    decoded_sentences = []
    target_texts_ =  []
    for seq_index in range(100):
        # Take one sequence (part of the training set)
        # for trying out decoding.

        input_seq = encoder_input_data[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq)
        target_text = target_texts[seq_index][1:-1]
        print('-')
        print('Input sentence:', input_texts[seq_index])
        print('GT sentence:', target_text)
        print('Decoded sentence:', decoded_sentence)   
        decoded_sentences.append(decoded_sentence)
        target_texts_.append(target_text)        
    #WER_spell_correction = calculate_WER(target_texts_, decoded_sentences)
    #print('WER_spell_correction: ', WER_spell_correction) 
    
def train_test_spell_corr_model(input_texts, target_texts)

    all_texts = target_texts + input_texts
    vocab_to_int, int_to_vocab = build_vocab(all_texts)

    input_characters = sorted(list(vocab_to_int))
    target_characters = sorted(list(vocab_to_int))
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    print('Number of samples:', len(input_texts))
    print('Number of unique input tokens:', num_encoder_tokens)
    print('Number of unique output tokens:', num_decoder_tokens)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)


    # Split the data into training and testing sentences
    input_texts, test_input_texts, target_texts, test_target_texts  = train_test_split(input_texts, target_texts, test_size = 0.15, random_state = 42)

    # ## Vectorize data

    # ## Train data
    encoder_input_data, decoder_input_data, decoder_target_data = vectorize_data(input_texts=input_texts,
                                                                                 target_texts=target_texts, 
                                                                                 max_encoder_seq_length=max_encoder_seq_length, 
                                                                                 num_encoder_tokens=num_encoder_tokens, 
                                                                                 vocab_to_int=vocab_to_int)


    # ## Test data
    test_encoder_input_data, test_decoder_input_data, test_decoder_target_data = vectorize_data(input_texts=test_input_texts,
                                                                                                target_texts=test_target_texts, 
                                                                                                max_encoder_seq_length=max_encoder_seq_length, 
                                                                                                num_encoder_tokens=num_encoder_tokens, 
                                                                                                vocab_to_int=vocab_to_int)


    # # Training model

    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    # TODO: Add Embedding for chars
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    print(model.summary())


    model.compile(optimizer=optimizers.Adam(lr=lr), loss='categorical_crossentropy', metrics=['categorical_accuracy'])


    #filepath="weights-improvement-{epoch:02d}-{val_categorical_accuracy:.2f}.hdf5"
    filepath="best_model.hdf5" # Save only the best model for inference step, as saving the epoch and metric might confuse the inference function which model to use
    checkpoint = ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')
    tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    callbacks_list = [checkpoint, tbCallBack]

    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              validation_data = ([test_encoder_input_data, test_decoder_input_data], test_decoder_target_data),
              batch_size=batch_size,
              epochs=epochs,
              callbacks=callbacks_list,
              #validation_split=0.2,
              shuffle=True)


    # # Inference model
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    # Sample output from train data
    decoded_sentences = []
    target_texts_ =  []
    for seq_index in range(100):
        # Take one sequence (part of the training set)
        # for trying out decoding.

        input_seq = encoder_input_data[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq)
        target_text = target_texts[seq_index][1:-1]
        print('-')
        print('Input sentence:', input_texts[seq_index])
        print('GT sentence:', target_text)
        print('Decoded sentence:', decoded_sentence)   
        decoded_sentences.append(decoded_sentence)
        target_texts_.append(target_text)



    #WER_spell_correction = calculate_WER(target_texts_, decoded_sentences)
    #print('WER_spell_correction |TRAIN= ', WER_spell_correction)


    # Sample output from test data
    # Sample output from train data
    decoded_sentences = []
    target_texts_ =  []
    for seq_index in range(100):
        # Take one sequence (part of the training set)
        # for trying out decoding.

        input_seq = test_encoder_input_data[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq)
        target_text = test_target_texts[seq_index][1:-1]
        print('-')
        print('Input sentence:', test_input_texts[seq_index])
        print('GT sentence:', target_text)
        print('Decoded sentence:', decoded_sentence)   
        decoded_sentences.append(decoded_sentence)
        target_texts_.append(target_text)


    WER_spell_correction = calculate_WER(test_target_texts_, decoded_sentences)
    print('WER_spell_correction |TEST= ', WER_spell_correction)





# # Results on merge of tesseract correction + generic data

# # Results on noisy tesseract corrections

# # Results noisy tesseract correction + generic data

# # Results of pre-training on generic and fine tuning on tesseract correction

# # Next steps
# - Add attention
# - Full attention
# - Condition the Encoder on word embeddings of the context (Bi-directional LSTM)
# - Condition the Decoder on word embeddings of the context (Bi-directional LSTM) 

# # References
# - Sequence to Sequence Learning with Neural Networks
#     https://arxiv.org/abs/1409.3215
# - Learning Phrase Representations using
#     RNN Encoder-Decoder for Statistical Machine Translation
#     https://arxiv.org/abs/1406.107
    
# # Load data
data_path = '..\'

max_sent_len = 40
min_sent_len = 4
num_samples = 10000
tess_correction_data = os.path.join(data_path, 'new_trained_data.txt')
input_texts_OCR, target_texts_OCR, gt_OCR = load_data_with_gt(tess_correction_data, num_samples, max_sent_len, min_sent_len)

input_texs = input_texs_OCR
target_texts = target_texts_OCR

train_test_spell_corr_model(input_texs, target_texts)

'''
# # Results of pre-training on generic data

# In[ ]:


num_samples = 10000
big_data = os.path.join(data_path, 'big.txt')
threshold = 0.9
input_texts_gen, target_texts_gen, gt_gen = load_data_with_noise(file_name=big_data, 
                                                                 num_samples=num_samples, 
                                                                 noise_threshold=threshold, 
                                                                 max_sent_len=max_sent_len, 
                                                                 min_sent_len=min_sent_len)
    
'''