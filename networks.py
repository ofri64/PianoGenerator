from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn

from transformer_layers import TransformerLayer


class SentimentNet(nn.Module):
    def __init__(self, n_vocab, hidden_dim=256, n_layers=2, drop_prob=0.5):
        super(SentimentNet, self).__init__()
        self.output_size = n_vocab
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = 250

        self.embedding = nn.Embedding(num_embeddings=n_vocab, embedding_dim=self.embedding_dim)
        self.dropout1 = nn.Dropout(drop_prob)
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim,
                            num_layers=n_layers, dropout=drop_prob, bidirectional=False)
        # self.dropout2 = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, n_vocab)

    def forward(self, x):
        batch_size, seq_length = x.size()
        x = x.view(batch_size, seq_length)

        x = self.embedding(x)
        x = self.dropout1(x)
        x = x.permute(1, 0, 2)  # Sequence, Batch, Features
        _, (h_n, _) = self.lstm(x)

        # take output of the last state (h_n) from the last LSTM layer (n_layers - 1)
        h_n = h_n.view(self.n_layers, 1, batch_size, self.hidden_dim)
        x = h_n[self.n_layers - 1, :, :, :]
        x = x.view(batch_size, self.hidden_dim)  # view as a flatten vector
        out = self.fc(x)
        return out


class TransformerNet(nn.Module):
    def __init__(self, embedding_dim, num_mappings, num_heads, num_blocks, seq_length, padding_idx=0):
        super().__init__()
        self.transformer = TransformerLayer(embedding_dim, num_mappings, num_heads, num_blocks, seq_length, padding_idx)
        self.linear = nn.Linear(in_features=embedding_dim, out_features=num_mappings)
        self.checkpoint_dir = None

    def forward(self, x):
        x = self.transformer(x)
        out = self.linear(x)

        return out

    def save_checkpoint(self, epoch, epoch_loss, optimizer_state_dict):
        if self.checkpoint_dir is None:
            self.checkpoint_dir = Path(f"training_checkpoint_{datetime.now().strftime('%d.%m.%Y_%H:%M:%S')}")
            self.checkpoint_dir.mkdir(parents=True)

        current_time = datetime.now().strftime('%d.%m.%Y_%H:%M:%S')
        checkpoint_path = self.checkpoint_dir / f"epoch_{epoch}{current_time}.pth"
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer_state_dict,
            "loss": epoch_loss
        }

        print(f"Saving model checkpoint for end of epoch {epoch}")
        torch.save(checkpoint, checkpoint_path)
