from transformers import AutoModel
import numpy as np
import soundfile as sf
import torch

print("Loading IndicF5...")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", device)

repo_id = "ai4bharat/IndicF5"

model = AutoModel.from_pretrained(
    repo_id,
    trust_remote_code=True
)

print("Model loaded.")

#text = """ನಮಸ್ಕಾರ ಸ್ನೇಹಿತರೇ. ಇವತ್ತಿನ ಪ್ರಮುಖ AI ಸುದ್ದಿಗೆ ಬರೋಣ.
#ಕೃತಕ ಬುದ್ಧಿಮತ್ತೆಯ ಕ್ಷೇತ್ರದಲ್ಲಿ ಇಂದು ಕೆಲವು ಪ್ರಮುಖ ಬೆಳವಣಿಗೆಗಳು ನಡೆದಿವೆ."""

with open("reference/script.txt",encoding="utf-8") as f:
	text = f.read().strip()

audio = model(
    text,
    ref_audio_path="reference/voice.wav",
    ref_text=open("reference/voice.txt", encoding="utf-8").read().strip()
)

if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0

audio = np.asarray(audio, dtype=np.float32)

sf.write(
    "output_kannada.wav",
    audio,
    samplerate=24000
)

print("Done!")
print("Saved: output_kannada.wav")
