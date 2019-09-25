""" This module prepares midi file data and feeds it to the neural
    network for training """
import torch
# from music21 import converter, instrument, note, chord

from training import train
from data_preprocess import load_training_data

random_seed = 0
torch.manual_seed(random_seed)

if __name__ == '__main__':
    data_path = "data/chopin/*.mid"
    seq_length = 256
    network_input, network_output, num_unique_tokens = load_training_data(data_dir_path=data_path, sequence_length=seq_length,
                                                                          save_data=False, load_data=True)
    train(network_input, network_output, num_unique_tokens, seq_length)
    # generation.generate()
