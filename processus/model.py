# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTM_CTC(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, num_layers=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.classifier(x)
        return x


class CNN_BiLSTM_CTC(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, num_layers=3, dropout=0.3):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.temporal_downsample_ratio = 2  
        self.lstm_input_dim = (input_dim // self.temporal_downsample_ratio) * 64

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # Classifieur final
        self.classifier = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1, 3).contiguous()
        B, T, C, F = x.shape
        x = x.view(B, T, C * F)
        x, _ = self.lstm(x)
        x = self.classifier(x)

        return x

    def compute_output_lengths(self, input_lengths):
        return input_lengths // self.temporal_downsample_ratio


class BiGRU_CTC(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, num_layers=3, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, bidirectional=True, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.classifier(x)
        return x

class LSTM_CTC(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, num_layers=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False, 
            dropout=dropout if num_layers > 1 else 0.0  # dropout uniquement si >1 couche
        )
        self.classifier = nn.Linear(hidden_dim, vocab_size)  # pas de *2 ici

    def forward(self, x):
        x, _ = self.lstm(x)  
        x = self.classifier(x)  
        return x


class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, bidirectional=True, dropout=dropout)

    def forward(self, x):
        outputs, hidden = self.gru(x)  # outputs: [B, T, 2*H], hidden: [2*L, B, H]
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 3, 1)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: [B, H]
        # encoder_outputs: [B, T, 2*H]
        seq_len = encoder_outputs.size(1)
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)  # [B, T, H]
        concat = torch.cat((decoder_hidden, encoder_outputs), dim=2)        # [B, T, 3H]
        energy = self.attn(concat).squeeze(2)                               # [B, T]
        attn_weights = F.softmax(energy, dim=1)                             # [B, T]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)     # [B, 1, 2H]
        return context.squeeze(1), attn_weights


class GRUDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers=1, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(hidden_dim * 2, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, context_vector, hidden):
        # context_vector: [B, 2H] → expand to [B, 1, 2H]
        input_step = context_vector.unsqueeze(1)
        output, hidden = self.gru(input_step, hidden)  # output: [B, 1, H]
        prediction = self.out(output.squeeze(1))       # [B, vocab_size]
        return prediction, hidden


class GRUSeq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, encoder_layers=2, decoder_layers=1):
        super().__init__()
        self.bridge = nn.Linear(hidden_dim * 2, hidden_dim)  # si encodeur bidirectionnel
        self.encoder_layers = encoder_layers  # ✅ AJOUTE CETTE LIGNE
        self.decoder_layers = decoder_layers  # ✅ AUSSI UTILE POUR LA SUITE
        self.hidden_dim = hidden_dim

        self.encoder = GRUEncoder(input_dim, hidden_dim, encoder_layers)
        self.attention = Attention(hidden_dim)
        self.decoder = GRUDecoder(hidden_dim, vocab_size, decoder_layers)

    def forward(self, x, max_len=100):
        # x : [B, T, input_dim]
        encoder_outputs, encoder_hidden = self.encoder(x)  # encoder_hidden: [2*L, B, H]
        batch_size = x.size(0)

        # Convertir [2*L, B, H] → [L, 2, B, H]
        encoder_hidden = encoder_hidden.view(self.encoder_layers, 2, batch_size, self.hidden_dim)

        # Concaténer directions forward/backward → [L, B, 2H]
        hidden_cat = torch.cat([encoder_hidden[:, 0], encoder_hidden[:, 1]], dim=-1)  # [L, B, 2H]

        # Projetter chaque couche via Linear → [L, B, H]
        if not hasattr(self, "bridge"):
            self.bridge = nn.Linear(self.hidden_dim * 2, self.hidden_dim).to(x.device)
        hidden_proj = self.bridge(hidden_cat)  # [L, B, H]

        # Adapter au nombre de couches attendues par le décodeur
        decoder_layers = self.decoder.gru.num_layers
        if hidden_proj.size(0) < decoder_layers:
            diff = decoder_layers - hidden_proj.size(0)
            repeat = hidden_proj[-1:, :, :].repeat(diff, 1, 1)
            hidden_proj = torch.cat([hidden_proj, repeat], dim=0)
        elif hidden_proj.size(0) > decoder_layers:
            hidden_proj = hidden_proj[:decoder_layers]

        # hidden_proj: [decoder_layers, B, H] — prêt pour le décodeur
        hidden = hidden_proj

        outputs = []
        for _ in range(max_len):
            context, attn_weights = self.attention(hidden[-1], encoder_outputs)  # [B, 2H]
            prediction, hidden = self.decoder(context, hidden)  # prediction: [B, vocab_size], hidden: [decoder_layers, B, H]
            outputs.append(prediction)

        return torch.stack(outputs, dim=1)  # [B, max_len, vocab_size]

