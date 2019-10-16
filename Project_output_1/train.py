"""
This module prepares midi file data and trains the
neural networks
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam
from keras import initializers

import glob
import pickle
from music21 import converter, instrument, note, chord
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

from keras.layers.wrappers import TimeDistributed
from keras.layers import concatenate

def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    for file in glob.glob("midi_songs/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([[np_utils.to_categorical(note_to_int[char], num_classes=n_vocab)] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    #network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    #network_input = network_input / float(n_vocab)
    """
    binary_network_input = []
    for i in network_input:
        binary_network_input.append(np_utils.to_categorical(i))

    print(binary_network_input[0].shape)
    print(type(binary_network_input[0]))
    
    for i in range(len(binary_network_input)):
        temp = binary_network_input[i]
        binary_network_input[i] = np.reshape(temp, (1, temp.shape[0], -1))

    print(binary_network_input[0].shape)

    binary_network_input_matrix = np.vstack([i for i in binary_network_input])
    

    return binary_network_input_matrix
    """
    print("#############################")
    print(len(network_input))
    print(len(network_input[0]))
    print(len(network_input[0][0]))
    print("#############################")
    network_input = np.reshape(network_input, (n_patterns, 100, n_vocab))
    return network_input
    
def generate_notes(generator, x_train, pitchnames, n_vocab):
    """ Generate notes from the neural network based on a sequence of notes """
    sequence_length = 100

    start = 0
    pattern = x_train[start]

    for _ in range(sequence_length):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = generator.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return pattern

def get_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)

def get_generator(input_shape, n_vocab, optimizer):
    generator = Sequential()
    
    generator.add(LSTM(
        512,
        input_shape=input_shape,
        return_sequences=True,
        kernel_initializer=initializers.RandomNormal(stddev=0.02)
    ))
    generator.add(Dropout(0.3))
    generator.add(LSTM(512, return_sequences=True))
    generator.add(Dropout(0.3))
    generator.add(LSTM(512, return_sequences=True))
    generator.add(TimeDistributed(Dense(256)))
    generator.add(Dropout(0.3))
    generator.add(TimeDistributed(Dense(input_shape[-1])))

    generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return generator

def get_discriminator(input_shape, n_vocab, optimizer):
    discriminator = Sequential()

    discriminator.add(LSTM(
        512,
        input_shape=input_shape,
        return_sequences=True,
        kernel_initializer=initializers.RandomNormal(stddev=0.02)
    ))
    discriminator.add(Dropout(0.3))
    discriminator.add(LSTM(512, return_sequences=True))
    discriminator.add(Dropout(0.3))
    discriminator.add(LSTM(512))
    discriminator.add(Dense(256))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(1, activation='softmax'))

    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return discriminator

def get_gan_network(discriminator, input_shape, generator, optimizer):
    # We initially set trainable to False since we only want to train either the 
    # generator or discriminator at a time
    discriminator.trainable = False
    sequence_length = 100
    # gan input (noise) will be 100-dimensional vectors
    gan_input = Input(shape=input_shape)
    # the output of the generator (a sequence)
    x = generator(gan_input)

    # get the output of the discriminator (probability if the image is real or not)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    return gan

def train(epochs=1, batch_size=128):
    notes = get_notes()
    pitchnames = sorted(set(item for item in notes))

    # Get all pitch names
    n_vocab = len(set(notes))

    # Get the training and testing data
    x_train = prepare_sequences(notes, n_vocab)

    
    # Split the training data into batches of size 128
    batch_count = x_train.shape[0] / batch_size

    input_shape = x_train.shape[0:]

    print("################################")
    print(x_train.shape)
    print(input_shape)
    print("################################")

    # Build our GAN netowrk
    adam = get_optimizer()
    generator = get_generator(input_shape, n_vocab, adam)
    discriminator = get_discriminator(input_shape, n_vocab, adam)
    gan = get_gan_network(discriminator, input_shape, generator, adam)

    for e in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in tqdm(range(int(batch_count))):
            # Get a random set of sequences
            sequences_batch = x_train[np.random.randint(0, x_train.shape[0], size=(batch_size))]

            # Generate fake sequences
            generated_sequences = generator.predict(sequences_batch)
            X = np.concatenate([sequences_batch, generated_sequences])

            # Labels for generated and real data
            y_dis = np.zeros(2*batch_size)
            # One-sided label smoothing
            y_dis[:batch_size] = 0.9

            # Train discriminator
            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)

            # Train generator
            sequences_batch = x_train[np.random.randint(0, x_train.shape[0], size=(batch_size))]
            generated_sequences = generator.predict(sequences_batch)
            
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            gan.train_on_batch(generated_sequences, y_gen)

        generator.save_weights("weights/weights_generator-" + str(e) + ".h5")
        discriminator.save_weights("weights/weights_discriminator-" + str(e) + ".h5")


if __name__ == '__main__':
    train(200, 128)