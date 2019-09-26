import glob
import pickle
from random import shuffle, seed

import numpy as np
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


def load_from_pickle(file_path):
    with open(file_path, 'rb') as filepath:
        data = pickle.load(filepath)
    return data


def save_to_pickle(file_path, data):
    with open(file_path, "wb") as filepath:
        pickle.dump(data, filepath, pickle.HIGHEST_PROTOCOL)


def prepare_sequences(notes, sequence_len, translator):
    """ Prepare the sequences used by the Neural Network """

    network_input = []
    network_labels = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_len - 1, 1):
        X = notes[i:i + sequence_len]
        y = notes[i+1: i+1 + sequence_len]
        network_input.append([translator[char] for char in X])
        network_labels.append([translator[char] for char in y])

    return network_input, network_labels


def prepare_predict_init_state(training_notes, sequence_len):
    start = np.random.randint(0, len(training_notes) - sequence_len -1)
    init_state = training_notes[start: start + sequence_len]
    return init_state


def load_training_data(data_dir_path, sequence_length=256, save_data=True, load_data=False):
    if load_data:
        training_data= load_from_pickle("training_data.pickle")
        all_notes = training_data["data"]
        note_translator = training_data["note_translator"]

    else:
        input_files = get_list_midi(data_dir_path)

        # get all training date samples
        all_notes = []
        for file_ in input_files:
            get_notes_from_midi_file(file_, all_notes)

        # get all pitch names
        unique_notes = set(item for item in all_notes)

        # create a dictionary to map pitches to integers
        note_translator = {note: index for index, note in enumerate(unique_notes)}

        if save_data:
            save_to_pickle("training_data.pickle", {
                "data": all_notes,
                "note_translator": note_translator
            })

    training_input, training_labels = prepare_sequences(all_notes, sequence_length, note_translator)
    num_unique_notes = len(note_translator)

    return training_input, training_labels, num_unique_notes
