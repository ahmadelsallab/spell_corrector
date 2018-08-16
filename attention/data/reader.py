import json
import csv
import random

import numpy as np
from keras.utils.np_utils import to_categorical

random.seed(1984)

INPUT_PADDING = 50
OUTPUT_PADDING = 100


class Vocabulary(object):

    def __init__(self, vocabulary_file=None, text_file=None, padding=None):
        """
            Creates a vocabulary from a file
            :param vocabulary_file: the path to the vocabulary
        """
        self.padding = padding
        
        # You can either load a ready vocab, or build it
        if(vocabulary_file):
            self.vocabulary_file = vocabulary_file
            with open(vocabulary_file, 'r') as f:
                self.vocabulary = json.load(f)

            
            self.reverse_vocabulary = {v: k for k, v in self.vocabulary.items()}

        else:              
            self.vocabulary, self.reverse_vocabulary = self.build_vocab(text_file)
            
    def build_vocab(self, text_file):
        '''Build vocab dictionary to victorize chars into ints'''
        vocab_to_int = {}
        count = 0
        #chars = open(text_file, encoding='utf8').read()
	chars = open(text_file).read()
    
        for char in chars:
            if char not in vocab_to_int:
                vocab_to_int[char] = count
                count += 1

        # Add special tokens to vocab_to_int
        codes = ['\t','\n', '<unk>', '<eos>']
        for code in codes:
            if code not in vocab_to_int:
                vocab_to_int[code] = count
                count += 1
        '''''Build inverse translation from int to char'''
        int_to_vocab = {}
        for character, value in vocab_to_int.items():
            int_to_vocab[value] = character
            
        return vocab_to_int, int_to_vocab    
        
    def size(self):
        """
            Gets the size of the vocabulary
        """
        return len(self.vocabulary.keys())

    def string_to_int(self, text):
        """
            Converts a string into it's character integer 
            representation
            :param text: text to convert
        """
        characters = list(text)

        integers = []

        if self.padding and len(characters) >= self.padding:
            # truncate if too long
            characters = characters[:self.padding - 1]

        characters.append('<eot>')

        for c in characters:
            if c in self.vocabulary:
                integers.append(self.vocabulary[c])
            else:
                integers.append(self.vocabulary['<unk>'])


        # pad:
        if self.padding and len(integers) < self.padding:
            integers.extend([self.vocabulary['<unk>']]
                            * (self.padding - len(integers)))

        if len(integers) != self.padding:
            print(text)
            raise AttributeError('Length of text was not padded.')
        return integers

    def int_to_string(self, integers):
        """
            Decodes a list of integers
            into it's string representation
        """
        characters = []
        for i in integers:
            characters.append(self.reverse_vocabulary[i])

        return characters


class Data(object):

    def __init__(self, file_name, input_vocabulary, output_vocabulary, delimiter='tab'):
        """
            Creates an object that gets data from a file
            :param file_name: name of the file to read from
            :param vocabulary: the Vocabulary object to use
            :param batch_size: the number of datapoints to return
            :param padding: the amount of padding to apply to 
                            a short string
        """

        self.input_vocabulary = input_vocabulary
        self.output_vocabulary = output_vocabulary
        self.file_name = file_name
        if delimiter == 'tab':
            self.delimiter = '\t'
        
    def load(self):
        """
            Loads data from a file
        """
        self.inputs = []
        self.targets = []

        with open(self.file_name, 'r') as f:
            #reader = csv.reader(f, delimiter='\t')
            #print(self.delimiter)
            reader = csv.reader(f, delimiter=self.delimiter)
            for row in reader:
                #print(row[0])
                self.inputs.append(row[0])
                #print(row[1])
                self.targets.append(row[1])

    def transform(self):
        """
            Transforms the data as necessary
        """
        # @TODO: use `pool.map_async` here?
        self.inputs = np.array(list(
            map(self.input_vocabulary.string_to_int, self.inputs)))
        self.targets = map(self.output_vocabulary.string_to_int, self.targets)
        self.targets = np.array(
            list(map(
                lambda x: to_categorical(
                    x,
                    num_classes=self.output_vocabulary.size()),
                self.targets)))

        assert len(self.inputs.shape) == 2, 'Inputs could not properly be encoded'
        assert len(self.targets.shape) == 3, 'Targets could not properly be encoded'

    def generator(self, batch_size):
        """
            Creates a generator that can be used in `model.fit_generator()`
            Batches are generated randomly.
            :param batch_size: the number of instances to include per batch
        """
        instance_id = range(len(self.inputs))
        while True:
            try:
                batch_ids = random.sample(instance_id, batch_size)
                yield (np.array(self.inputs[batch_ids], dtype=int),
                       np.array(self.targets[batch_ids]))
            except Exception as e:
                print('EXCEPTION OMG')
                print(e)
                yield None, None

'''                
if __name__ == '__main__':

    input_vocab = Vocabulary('./human_vocab.json', padding=50)
    output_vocab = Vocabulary('./machine_vocab.json', padding=12)
    ds = Data('./fake.csv', input_vocab, output_vocab)
    ds.load()
    ds.transform()
    print(ds.inputs.shape)
    print(ds.targets.shape)
    g = ds.generator(32)
    print(ds.inputs[[5,10, 12]].shape)
    print(ds.targets[[5,10,12]].shape)
    # for i in range(50):
    #     print(next(g)[0].shape)
    #     print(next(g)[1].shape)
'''
