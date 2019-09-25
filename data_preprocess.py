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
