````markdown
# 🔉 Reconnaissance Automatique de la Parole en Yemba (ASR-Yemba)

Ce projet propose une **application de reconnaissance vocale automatique (ASR)** pour la langue **Yemba**, une langue tonale du Cameroun. Il s’appuie sur une architecture **GRU Seq2Seq avec mécanisme d’attention additive**, permettant la transcription syllabique et tonale à partir d’échantillons audio.

## 🧠 Objectif du projet

Développer un modèle capable de transcrire automatiquement les énoncés oraux en Yemba en intégrant :
- La **structure syllabique**
- Les **variations tonales (haut, moyen, bas)**
- Un **vocabulaire personnalisé** issu du corpus *YembaTones*

---

## 🗃️ Données utilisées

Le projet utilise le corpus [YembaTones](https://data.mendeley.com/datasets/cx268tmrwn/3) :
- 344 mots en Yemba, enregistrés par 11 locuteurs natifs.
- Fichiers `.wav` + annotations `.TextGrid` (segmentations syllabiques et tons).
- Format unifié de transcription : `syllabe|ton syllabe|ton ...`

---

## 🧱 Architecture du modèle

L’architecture implémentée se compose de :

| Module        | Description |
|---------------|-------------|
| `GRUEncoder`  | GRU bidirectionnel pour encoder les melspectrogrammes. |
| `Attention`   | Mécanisme de Bahdanau pour le focus dynamique. |
| `GRUDecoder`  | GRU unidirectionnel pour générer la transcription. |
| `GRUSeq2Seq`  | Intégration complète de l’encodeur, attention et décodeur. |

Le modèle final est défini dans [`model.py`](./model.py). Plusieurs variantes sont incluses : BiLSTM-CTC, BiGRU-CTC, CNN-BiLSTM, etc.

---

## 🔁 Pipeline de traitement

1. **Prétraitement audio** (Mono 16kHz) → extraction des **Melspectrogrammes**.
2. **Tokenisation** des transcriptions syllabico-tonales.
3. **Division du jeu de données** (80% train / 10% val / 10% test).
4. **Entraînement** avec `CrossEntropyLoss`, `Teacher Forcing`, et `EarlyStopping`.
5. **Évaluation** avec les métriques :
   - WER (Word Error Rate)
   - CER (Character Error Rate)
   - Précision brute
6. **Déploiement** via interface [Gradio](https://www.gradio.app).

---

## 💻 Interface Utilisateur

Une application web simple construite avec **Gradio** permet :
- Chargement d’un fichier `.wav`
- Transcription brute avec les tons (`pá|haut`)
- Transcription nettoyée pour la lecture (`pá`)

➡️ [Accès temporaire en ligne](https://91465d951e86d66f8a.gradio.live/)

---

## 📊 Résultats

| Indicateur | Valeur |
|------------|--------|
| **WER**    | 63.02% |
| **CER**    | 42.79% |
| **Précision brute** | ~8.63% |

---

## 📦 Dépendances principales

```bash
torch >= 2.x
torchaudio
numpy, pandas
matplotlib
gradio
jiwer
soundfile
````

📁 Vérifiez [`test.py`](./test.py) pour les formats audio supportés via `soundfile`.

---

## ⚖️ Considérations éthiques

* Respect des données personnelles : pas de collecte utilisateur.
* Corpus anonymisé et ouvert (licence académique).
* Code source transparent et reproductible.
* Biais reconnus et décrits (âge, zone géographique, style de parole).
* Finalité strictement éducative, patrimoniale et scientifique.

---

## 🚀 Lancer l'application en local

```bash
# Installation
pip install -r requirements.txt

# Exécution locale
python interface.py  # ou autre script Gradio d'interface
```

---

## 👥 Équipe du projet

* **Noubissi Fopa Christian Junior**
* **Essuthi Mbangue Ange Armel** 
* **Nguemtchueng Tsamo Bibiane Danielle** 
* **Moukeki Indjandja Dave Kevin** 
* **Abanda Armand Wilfried** 

**Supervision : Pr. Paulin MALETAGIA**

---

## 📚 Références clés

* [YembaTones Corpus](https://data.mendeley.com/datasets/cx268tmrwn/3)
* [PyTorch](https://pytorch.org)
* [Gradio](https://www.gradio.app/)
* [Hugging Face ASR Models](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition)
* [DeepSpeech Mandarin](https://github.com/PaddlePaddle/DeepSpeech)

```

