import torch
import torch.nn as nn
from torch.utils import data

from datasets import Dataset
from networks import TransformerNet


def train(training_input, training_labels, num_unique_tokens, sequence_length):
    # Loader Parameters
    params = {'batch_size': 32,
              'shuffle': True,
              'num_workers': 12}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create Batch Generator
    training_dataset = Dataset(training_input, training_labels)
    training_dataloader = torch.utils.data.DataLoader(training_dataset, **params)

    # Define network, loss function and optimizer
    net = TransformerNet(embedding_dim=500, num_mappings=num_unique_tokens,
                         num_heads=4, num_blocks=3, seq_length=sequence_length)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-3)

    # Define number of epochs
    num_epochs = 10
    print_batch_step = 100
    best_epoch_loss = 1000

    # Start training
    net = net.to(device)
    net.train(mode=True)
    for epoch in range(num_epochs):

        running_loss = 0
        epoch_loss = 0

        for i, data_ in enumerate(training_dataloader, 1):

            inputs, labels = data_[0].to(device), data_[1].to(device)

            optimizer.zero_grad()
            outputs = net(inputs)

            # flatten sequence dim to batch dim before Softmax + CE
            batch, seq, num_classes = outputs.size()
            outputs = outputs.view(batch * seq, num_classes)
            labels = labels.view(batch * seq)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_loss += loss.item() * params['batch_size']
            if i % print_batch_step == 0:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i, running_loss / print_batch_step))
                running_loss = 0

        epoch_num = epoch + 1
        average_epoch_loss = epoch_loss / len(training_dataset)
        print(f"Epoch {epoch_num}: average loss is {average_epoch_loss}")

        if average_epoch_loss < best_epoch_loss:
            net.save_checkpoint(epoch_num, average_epoch_loss, optimizer.state_dict())
            best_epoch_loss = average_epoch_loss

    print('Finished Training')
