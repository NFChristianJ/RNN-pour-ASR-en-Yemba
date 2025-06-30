import gradio as gr
import torch
import torchaudio
import json
from model import GRUSeq2Seq
torchaudio.set_audio_backend("soundfile")

# Chargement du modèle et vocabulaire
MODEL_PATH = "best_model_GRUseq2seq.pt"
VOCAB_PATH = "vocab4.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_MELS = 80
HIDDEN_DIM = 256
ENC_LAYERS = 2
DEC_LAYERS = 2

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = json.load(f)
vocab = {k: int(v) for k, v in vocab.items()}
vocab_inv = {v: k for k, v in vocab.items()}
PAD_IDX = vocab.get("<pad>", 0)

model = GRUSeq2Seq(input_dim=N_MELS, hidden_dim=HIDDEN_DIM,
                   vocab_size=len(vocab),
                   encoder_layers=ENC_LAYERS,
                   decoder_layers=DEC_LAYERS).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Fonctions
def preprocess_audio(file_path):
    waveform, sr = torchaudio.load(file_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=256, n_mels=N_MELS)(waveform)
    mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    return mel_spec.squeeze(0).transpose(0, 1)

def predict(file_path):
    mel_spec = preprocess_audio(file_path).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(mel_spec)
        probs = logits.softmax(2)
        preds = probs.argmax(2)[0]
    output_tokens = []
    prev = -1
    for p in preds:
        idx = p.item()
        if idx != prev and idx != PAD_IDX:
            output_tokens.append(idx)
        prev = idx
    return " ".join(vocab_inv.get(i, "") for i in output_tokens)

def predict_cleaned(raw_pred):
    tokens = raw_pred.split()
    cleaned = []
    for token in tokens:
        if token.strip() == "" or token.startswith("∅"):
            continue
        base = token.split("|")[0].replace("'", "")
        cleaned.append(base)
    return "".join(cleaned)

# Fonction pour Gradio 
def transcribe_from_file(file_obj):
    if not file_obj:
        return None, "Aucun fichier", ""
    audio_path = file_obj.name
    raw = predict(audio_path)
    clean = predict_cleaned(raw)
    return audio_path, raw, clean


# Interface Gradio
demo = gr.Interface(
    fn=transcribe_from_file,
    inputs=gr.File(label="Charger un fichier .wav (16kHz mono)"),
    outputs=[
        gr.Audio(label="🎧 Lecture de l’audio"),
        gr.Textbox(label="Transcription brute (avec tonalités)"),
        gr.Textbox(label="Transcription nettoyée (syllabes)"),
    ],
    title="🗣️ Reconnaissance Vocale Yemba",
    description="Transcrivez un fichier .wav (mono 16kHz) et écoutez l'audio directement."
)

if __name__ == "__main__":
    demo.launch(share=True)
