import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super().__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        self.rnn1 = nn.LSTM(n_features, self.hidden_dim, batch_first=True)
        self.rnn2 = nn.LSTM(self.hidden_dim, embedding_dim, batch_first=True)

    def forward(self, x):
        # x shape: (seq_len, n_features) or (batch, seq_len, n_features)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # add batch dimension
        x, _ = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        return hidden_n.reshape((x.size(0), self.n_features, self.embedding_dim))


class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super().__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features

        self.rnn1 = nn.LSTM(input_dim, input_dim, batch_first=True)
        self.rnn2 = nn.LSTM(input_dim, self.hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        # repeat embedding across sequence length
        x = x.repeat(1, self.seq_len, 1)  # (batch, seq_len, input_dim)
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        return self.output_layer(x)


class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super().__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim)
        self.decoder = Decoder(seq_len, embedding_dim, n_features)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
