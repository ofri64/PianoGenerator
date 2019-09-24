import glob
import pickle
from random import shuffle, seed

from music21 import converter, instrument, note, chord


def get_list_midi(folder, seed_int=666):
    """Get the list of all midi file in the folders

    Parameters
    ==========
    folder : str
      The midi folder.
    seed_int : int
      the random seed.

    Returns
    =======
    The midi files

    """
    list_all_midi = glob.glob(folder)
    seed(seed_int)
    shuffle(list_all_midi)
    return list_all_midi


def get_notes_from_midi_file(file_path, notes_list):
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    midi = converter.parse(file_path)

    try:  # file has instrument parts
        s2 = instrument.partitionByInstrument(midi)
        notes_to_parse = s2.parts[0].recurse()
    except:  # file has notes in a flat structure
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes_list.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes_list.append('.'.join(str(n) for n in element.normalOrder))


def load_notes_from_pickle(file_path):
    with open(file_path, 'rb') as filepath:
        notes = pickle.load(filepath)
    return notes


def save_notes_to_pickle(file_path, notes):
    with open(file_path, "wb") as filepath:
        pickle.dump(notes, filepath, pickle.HIGHEST_PROTOCOL)


def prepare_sequences(notes, sequence_len):
    """ Prepare the sequences used by the Neural Network """

    # get all pitch names
    unique_notes = set(item for item in notes)

    # create a dictionary to map pitches to integers
    note_to_int = {note: index for index, note in enumerate(unique_notes)}

    network_input = []
    network_labels = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_len - 1, 1):
        X = notes[i:i + sequence_len]
        y = notes[i+1: i+1 + sequence_len]
        network_input.append([note_to_int[char] for char in X])
        network_labels.append([note_to_int[char] for char in y])

    return network_input, network_labels, len(unique_notes)


def load_training_data(data_dir_path, sequence_length=256, save_data=True, load_data=False):
    if load_data:
        all_notes = load_notes_from_pickle("all_notes.pickle")

    else:
        input_files = get_list_midi(data_dir_path)
        all_notes = []
        for file_ in input_files:
            get_notes_from_midi_file(file_, all_notes)

        if save_data:
            save_notes_to_pickle("all_notes.pickle", all_notes)

    training_input, training_labels, num_unique_notes = prepare_sequences(all_notes, sequence_len=sequence_length)

    return training_input, training_labels, num_unique_notes

# def to_categorical(y, num_classes=None, dtype='float32'):
#     """Converts a class vector (integers) to binary class matrix.
#     E.g. for use with categorical_crossentropy.
#     # Arguments
#         y: class vector to be converted into a matrix
#             (integers from 0 to num_classes).
#         num_classes: total number of classes.
#         dtype: The data type expected by the input, as a string
#             (`float32`, `float64`, `int32`...)
#     # Returns
#         A binary matrix representation of the input. The classes axis
#         is placed last.
#     # Example
#     ```python
#     # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
#     > labels
#     array([0, 2, 1, 2, 0])
#     # `to_categorical` converts this into a matrix with as many
#     # columns as there are classes. The number of rows
#     # stays the same.
#     > to_categorical(labels)
#     array([[ 1.,  0.,  0.],
#            [ 0.,  0.,  1.],
#            [ 0.,  1.,  0.],
#            [ 0.,  0.,  1.],
#            [ 1.,  0.,  0.]], dtype=float32)
#     ```
#     """
#
#     y = np.array(y, dtype='int')
#     input_shape = y.shape
#     if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
#         input_shape = tuple(input_shape[:-1])
#     y = y.ravel()
#     if not num_classes:
#         num_classes = np.max(y) + 1
#     n = y.shape[0]
#     categorical = np.zeros((n, num_classes), dtype=dtype)
#     categorical[np.arange(n), y] = 1
#     output_shape = input_shape + (num_classes,)
#     categorical = np.reshape(categorical, output_shape)
#     return categorical
#
#
# def normalize(x, axis=-1, order=2):
#     """Normalizes a Numpy array.
#     # Arguments
#         x: Numpy array to normalize.
#         axis: axis along which to normalize.
#         order: Normalization order (e.g. 2 for L2 norm).
#     # Returns
#         A normalized copy of the array.
#     """
#     l2 = np.atleast_1d(np.linalg.norm(x, order, axis))
#     l2[l2 == 0] = 1
#     return x / np.expand_dims(l2, axis)