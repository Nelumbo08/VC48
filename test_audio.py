import torch
import torchaudio

print(f"PyTorch version: {torch.__version__}")
print(f"Torchaudio version: {torchaudio.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Create a simple spectrogram transform
transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=48000,
    n_fft=1920,
    hop_length=480,
    n_mels=128
).cuda()

# Create a random audio signal
audio = torch.randn(1, 48000).cuda()

# Try to compute spectrogram
try:
    spec = transform(audio)
    print("\nSuccessfully computed mel spectrogram with shape:", spec.shape)
    print("Test passed!")
except Exception as e:
    print("\nError occurred:", str(e)) 