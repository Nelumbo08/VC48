import os
import torch
import torchaudio
#import webrtcvad
from torch.utils.data import Dataset, DataLoader
import os
import torchaudio.functional as F
import torch.nn.functional as Fnn
from torchaudio.sox_effects import apply_effects_tensor
from torchaudio.transforms import MelSpectrogram




class WaveformDataset(Dataset):
    def __init__(self, data_dir, sample_rate=48000, chunk_size=24000):
        """
        Args:
            data_dir: Directory containing audio files
            sample_rate: Target sample rate
            chunk_size: Size of audio chunks in samples (25ms * sample_rate)
        """
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
        
        # Pre-load and chunk all audio files
        self.chunks = []
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1920,
            hop_length=480,
            n_mels=128,
            power=1.0,
            normalized=True,
            mel_scale='slaney',
            win_length=1920,
            norm = 'slaney'
        )
        
        for file in self.files:
            waveform, sr = torchaudio.load(os.path.join(data_dir, file))
            if sr != sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
            
            # Convert to mono if stereo
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Split into chunks
            num_chunks = waveform.size(1) // chunk_size
            for i in range(num_chunks):
                chunk = waveform[:, i*chunk_size:(i+1)*chunk_size]
                if torch.isnan(chunk).any():
                    print(f"NaN found in chunk {i}")
                if chunk.size(1) == chunk_size:  # Only keep complete chunks
                    self.chunks.append(chunk)
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        # Compute mel spectrogram
        mel_spec = self.mel_transform(chunk)
        return {
            'waveform': chunk.squeeze(0),
            'mel_spec': mel_spec.squeeze(0)
        }
    

class Wav2Mel(torch.nn.Module):
    """Transform audio file into mel spectrogram tensors."""

    def __init__(
        self,
        sample_rate: float = 48000,
        norm_db: float = -3.0,
        sil_threshold: float = 10.0,
        sil_duration: float = 0.4,
        fft_window_ms: float = 40.0,
        fft_hop_ms: float = 10.0,
        n_fft: int = 1920,
        f_min: float = 50.0,
        n_mels: int = 128,
        preemph: float = 0.95,
        ref_db: float = 20.0,
        dc_db: float = 100.0,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.norm_db = norm_db
        self.sil_threshold = sil_threshold
        self.sil_duration = sil_duration
        self.fft_window_ms = fft_window_ms
        self.fft_hop_ms = fft_hop_ms
        self.n_fft = n_fft
        self.f_min = f_min
        self.n_mels = n_mels
        self.preemph = preemph
        self.ref_db = ref_db
        self.dc_db = dc_db

        #self.sox_effects = SoxEffects(sample_rate, norm_db, sil_threshold, sil_duration)
        self.log_melspectrogram = LogMelspectrogram(
            sample_rate,
            fft_window_ms,
            fft_hop_ms,
            n_fft,
            f_min,
            n_mels,
            preemph,
            ref_db,
            dc_db,
        )

    def forward(self, wav_tensor: torch.Tensor, sample_rate: int) -> torch.Tensor:
        #wav_tensor = self.sox_effects(wav_tensor, sample_rate)
        mel_tensor = self.log_melspectrogram(wav_tensor)
        return mel_tensor


class SoxEffects(torch.nn.Module):
    """Transform waveform tensors."""

    def __init__(
        self,
        sample_rate: int,
        norm_db: float,
        sil_threshold: float,
        sil_duration: float,
    ):
        super().__init__()
        self.effects = [
            ["channels", "1"],  # convert to mono
            ["rate", f"{sample_rate}"],  # resample
            ["norm", f"{norm_db}"],  # normalize to -3 dB
            [
                "silence",
                "1",
                f"{sil_duration}",
                f"{sil_threshold}%",
                "-1",
                f"{sil_duration}",
                f"{sil_threshold}%",
            ],  # remove silence throughout the file
        ]

    def forward(self, wav_tensor: torch.Tensor, sample_rate: int) -> torch.Tensor:
        wav_tensor, _ = apply_effects_tensor(wav_tensor, sample_rate, self.effects)
        if wav_tensor.numel() == 0:
            pass
        else:
        	return wav_tensor


class LogMelspectrogram(torch.nn.Module):
    """Transform waveform tensors into log mel spectrogram tensors."""

    def __init__(
        self,
        sample_rate: float,
        fft_window_ms: float,
        fft_hop_ms: float,
        n_fft: int,
        f_min: float,
        n_mels: int,
        preemph: float,
        ref_db: float,
        dc_db: float,
    ):
        super().__init__()
        self.melspectrogram = MelSpectrogram(
            sample_rate=sample_rate,
            win_length=int(sample_rate * fft_window_ms / 1000),
            hop_length=int(sample_rate * fft_hop_ms / 1000),
            n_fft=n_fft,
            f_min=f_min,
            n_mels=n_mels,
            power=1.0,
        )
        self.preemph = preemph
        self.ref_db = ref_db
        self.dc_db = dc_db

    def forward(self, wav_tensor: torch.Tensor) -> torch.Tensor:
        # preemph
        wav_tensor = torch.cat(
            (
                wav_tensor[:, 0].unsqueeze(-1),
                wav_tensor[:, 1:] - self.preemph * wav_tensor[:, :-1],
            ),
            dim=-1,
        )
        mel_tensor = self.melspectrogram(wav_tensor).squeeze(0).T  # (time, n_mels)
        mel_tensor = 20 * mel_tensor.clamp(min=1e-9).log10()
        mel_tensor = (mel_tensor - self.ref_db + self.dc_db) / self.dc_db
        return mel_tensor