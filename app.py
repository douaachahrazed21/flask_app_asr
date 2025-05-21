from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import difflib
import os
import requests

app = Flask(__name__)
CORS(app)

# Charger le modèle Whisper
processor = WhisperProcessor.from_pretrained("moatazlumin/Arabic_ASR_whisper_small_with_diacritics")
model = WhisperForConditionalGeneration.from_pretrained("moatazlumin/Arabic_ASR_whisper_small_with_diacritics")
model.eval()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_correct_quran_text(sura=1, aya=1):
    url = f"https://api.alquran.cloud/v1/ayah/{sura}:{aya}/ar"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['data']['verses'][0]['text_uthmani']
    return ""

def compare_texts(reference, hypothesis):
    sequence = difflib.SequenceMatcher(None, reference, hypothesis)
    result = ""
    for opcode, i1, i2, j1, j2 in sequence.get_opcodes():
        if opcode == 'equal':
            result += reference[i1:i2]
        elif opcode in ('replace', 'insert', 'delete'):
            result += f"<span style='color:red'>{reference[i1:i2]}</span>"
    return result

def calculate_similarity(reference, hypothesis):
    return difflib.SequenceMatcher(None, reference, hypothesis).ratio() * 100

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    filename = secure_filename(audio_file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    audio_file.save(filepath)

    waveform, sample_rate = torchaudio.load(filepath)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    input_features = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_features

    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    reference_text = get_correct_quran_text(sura=1, aya=1)

    similarity = calculate_similarity(reference_text, transcription)

    if similarity >= 95:
        return jsonify({
            "status": "correct",
            "text": transcription,
            "accuracy": similarity
        })
    else:
        highlighted = compare_texts(reference_text, transcription)
        return jsonify({
            "status": "incorrect",
            "text": highlighted,
            "accuracy": similarity
        })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=True)

