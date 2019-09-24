""" This module prepares midi file data and feeds it to the neural
    network for training """
import torch
# from music21 import converter, instrument, note, chord

from training import train
from data_preprocess import get_list_midi, load_training_data

random_seed = 0
torch.manual_seed(random_seed)


# def train_network():
#     """ Train a Neural Network to generate music """
#     # notes = get_notes()
#     # with open('all_notes', "wb") as filepath:
#     #     pickle.dump(notes, filepath, pickle.HIGHEST_PROTOCOL)
#
#     with open('all_notes', 'rb') as filepath:
#         notes = pickle.load(filepath)
#
#     # get amount of pitch names
#     n_vocab = len(set(notes))
#
#     network_input, network_output = prepare_sequences(notes)
#
#     model = SentimentNet(n_vocab)  # sending later to gpu memory
#     train(model, network_input, network_output)
#     return model


# def get_notes():
#     """ Get all the notes and chords from the midi files in the ./midi_songs directory """
#     notes = []
#
#     for file in sampled_200_midi:  # glob.glob('maestro-v1.0.0/**/*.midi'):
#         print("Parsing %s" % file)
#         midi = converter.parse(file)
#
#         notes_to_parse = None
#
#         try:  # file has instrument parts
#             s2 = instrument.partitionByInstrument(midi)
#             notes_to_parse = s2.parts[0].recurse()
#         except:  # file has notes in a flat structure
#             notes_to_parse = midi.flat.notes
#
#         for element in notes_to_parse:
#             if isinstance(element, note.Note):
#                 notes.append(str(element.pitch))
#             elif isinstance(element, chord.Chord):
#                 notes.append('.'.join(str(n) for n in element.normalOrder))
#
#     with open('notes', 'wb') as filepath:
#         pickle.dump(notes, filepath)
#
#     return notes
#
#
# def prepare_sequences(notes):
#     """ Prepare the sequences used by the Neural Network """
#     sequence_length = 100
#
#     # get all pitch names
#     pitchnames = sorted(set(item for item in notes))
#
#     # create a dictionary to map pitches to integers
#     note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
#
#     network_input = []
#     network_output = []
#
#     # create input sequences and the corresponding outputs
#     for i in range(0, len(notes) - sequence_length, 1):
#         sequence_in = notes[i:i + sequence_length]
#         sequence_out = notes[i + sequence_length]
#         network_input.append([note_to_int[char] for char in sequence_in])
#         network_output.append(note_to_int[sequence_out])
#
#     n_patterns = len(network_input)
#
#     # reshape the input into a format compatible with LSTM layers
#     network_input = numpy.reshape(network_input, (n_patterns, sequence_length))
#     # normalize input
#     #### this is not a normalization
#     #### also you need to treat the input as discrete tokens and perform embedding before using a recurrent layer
#     # network_input = network_input / float(n_vocab)
#
#     # network_output = to_categorical(network_output)  # np_utils.to_categorical(network_output)
#
#     return network_input, network_output


if __name__ == '__main__':
    data_path = "data/chopin/*.mid"
    seq_length = 256
    network_input, network_output, num_unique_tokens = load_training_data(data_dir_path=data_path, sequence_length=seq_length,
                                                                          save_data=False, load_data=True)
    train(network_input, network_output, num_unique_tokens, seq_length)
    # generation.generate()
