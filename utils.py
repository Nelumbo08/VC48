import torch
import torchaudio
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math
import torchaudio
import numpy as np
from scipy.signal import get_window
from models import iSTFT_generator


def load_vocoder():
    vocoder = iSTFT_generator().cuda()
    model = torch.load('/home/goquest/Voc48/checkpoints/vocoder_v3/vocoder_gan_1000.pth',weights_only=True)
    vocoder.load_state_dict(model)
    vocoder.eval()
    return vocoder



def load_audio(file_path, sample_rate=48000):
    """Load an audio file and resample it."""
    waveform, sr = torchaudio.load(file_path)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    return waveform

def get_mel_spectrogram(waveform, n_fft=1024, hop_length=256, n_mels=80):
    """Convert waveform to mel spectrogram."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=22050,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        center=True,
        power=1.0,
    )
    mel_spec = mel_transform(waveform)
    return torch.log(torch.clamp(mel_spec, min=1e-5))

def extract_f0(waveform, sample_rate=48000, hop_length=480):
    """Extract F0 (fundamental frequency) from waveform."""
    f0 = torchaudio.functional.detect_pitch_frequency(
        waveform, 
        sample_rate=sample_rate,
        frame_time=hop_length/sample_rate
    )
    return torch.log(torch.clamp(f0, min=1e-5))


class TorchSTFT(torch.nn.Module):
    def __init__(self, filter_length=1920, hop_length=480, win_length=1920, window='hann', device="cuda"):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.from_numpy(get_window(window, win_length, fftbins=True).astype(np.float32))
        self.device = device

    def transform(self, input_data):
        forward_transform = torch.stft(
            input_data,
            self.filter_length, self.hop_length, self.win_length, window=self.window.to(input_data.device()),
            return_complex=True)

        return torch.abs(forward_transform), torch.angle(forward_transform)

    def inverse(self, magnitude, phase):
        magnitude, phase = magnitude.to(self.device), phase.to(self.device)
        inverse_transform = torch.istft(
            magnitude * torch.exp(phase * 1j),
            self.filter_length, self.hop_length, self.win_length, window=self.window.to(self.device))

        return inverse_transform.unsqueeze(-2)  # unsqueeze to stay consistent with conv_transpose1d implementation

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction



class LossCalculator:
    def __init__(self, lambda_adv=1.0, lambda_perc=1.0, lambda_cycle=5.0,
                 lambda_id=5.0, lambda_style=1.0, lambda_f0=1.0, lambda_embed=0.1):
        self.lambda_adv = lambda_adv
        self.lambda_perc = lambda_perc
        self.lambda_cycle = lambda_cycle
        self.lambda_id = lambda_id
        self.lambda_style = lambda_style
        self.lambda_f0 = lambda_f0
        self.lambda_embed = lambda_embed
        
    def adversarial_loss(self, disc_outputs, target_is_real):
        """Hinge loss for adversarial training."""
        if target_is_real:
            return -torch.mean(torch.min(disc_outputs - 1, torch.zeros_like(disc_outputs)))
        else:
            return -torch.mean(torch.min(-disc_outputs - 1, torch.zeros_like(disc_outputs)))
    
    def perceptual_loss(self, real_feats, fake_feats):
        """L1 loss between real and generated feature maps."""
        losses = []
        for real_feat, fake_feat in zip(real_feats, fake_feats):
            losses.append(F.l1_loss(real_feat, fake_feat))
        return torch.mean(torch.stack(losses))
    
    def cycle_consistency_loss(self, real, cycled):
        """L1 loss between original and reconstructed spectrograms."""
        return F.l1_loss(real, cycled)
    
    def identity_loss(self, real, identity):
        """L1 loss for identity mapping."""
        return F.l1_loss(real, identity)
    
    def style_loss(self, real_feats, fake_feats):
        """Gram matrix based style loss."""
        def gram_matrix(feat):
            b, c, h, w = feat.size()
            feat = feat.view(b, c, -1)
            return torch.bmm(feat, feat.transpose(1, 2)) / (c * h * w)
            
        style_loss = 0
        for real_feat, fake_feat in zip(real_feats, fake_feats):
            gram_real = gram_matrix(real_feat)
            gram_fake = gram_matrix(fake_feat)
            style_loss += F.mse_loss(gram_real, gram_fake)
        return style_loss
    
    def f0_loss(self, real_f0, fake_f0):
        """RMSE loss for F0 contours."""
        return torch.sqrt(F.mse_loss(real_f0, fake_f0))
    
    def embedding_loss(self, source_embed, target_embed):
        """KL divergence between source and target embeddings."""
        return F.kl_div(
            F.log_softmax(source_embed, dim=-1),
            F.softmax(target_embed, dim=-1),
            reduction='batchmean'
        )
    
    def generator_loss(self, real_A, fake_B, cycled_A, identity_B,
                      disc_fake_B, real_feats_B, fake_feats_B,
                      source_f0, fake_f0, source_embed, target_embed):
        """Calculate total generator loss."""
        loss_adv = self.adversarial_loss(disc_fake_B, True) * self.lambda_adv
        loss_perc = self.perceptual_loss(real_feats_B, fake_feats_B) * self.lambda_perc
        loss_cycle = self.cycle_consistency_loss(real_A, cycled_A) * self.lambda_cycle
        loss_id = self.identity_loss(real_A, identity_B) * self.lambda_id
        loss_style = self.style_loss(real_feats_B, fake_feats_B) * self.lambda_style
        loss_f0 = self.f0_loss(source_f0, fake_f0) * self.lambda_f0
        loss_embed = self.embedding_loss(source_embed, target_embed) * self.lambda_embed
        
        total_loss = (loss_adv + loss_perc + loss_cycle + 
                     loss_id + loss_style + loss_f0 + loss_embed)
        
        return {
            'total': total_loss,
            'adv': loss_adv,
            'perc': loss_perc,
            'cycle': loss_cycle,
            'id': loss_id,
            'style': loss_style,
            'f0': loss_f0,
            'embed': loss_embed
        }
    
    def discriminator_loss(self, disc_real, disc_fake):
        """Calculate discriminator loss."""
        loss_real = self.adversarial_loss(disc_real, True)
        loss_fake = self.adversarial_loss(disc_fake, False)
        return (loss_real + loss_fake) * 0.5

def save_checkpoint(model, optimizer, epoch, path):
    """Save model and optimizer state."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def load_checkpoint(model, optimizer, path):
    """Load model and optimizer state."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']


# Utility functions

def kl_divergence_loss(V, V_hat):
    return torch.sum(V * torch.log((V + 1e-10) / (V_hat + 1e-10)) - V + V_hat)

def mse_loss(V, V_hat):
    return nn.MSELoss()(V, V_hat)

def SpecLoss(m1,m2):
    m1=m1#.detach().cpu().numpy()
    m2=m2
    #sl=torch.abs(torch.log(torch.abs(torch.from_numpy(dct(m1))+0.0000001))-torch.log(torch.abs(torch.from_numpy(dct(m2))+0.0000001)))
    sl=torch.abs(dct1(m1)-dct1(m2))
    #sll=torch.from_numpy(sl)
    return sl

def reconstruct(H, W):
        pad_size = (W.shape[2] - 1, W.shape[3] - 1)
        out = torch.nn.functional.conv2d(H, W.flip((2, 3)), padding=pad_size)
        return out


def dct1(x, type=2, norm='ortho'):

    dct_func = {
        1: torch.fft.fft,
        2: torch.fft.rfft,
        3: torch.fft.rfft,
    }[type]
    
    # Compute DFT along the last dimension
    dft_x = dct_func(x, dim=-1)
    
    # Select appropriate components for the DCT
    if type == 1:
        dft_x = dft_x[..., 0]  # use only the real part
    elif type == 2:
        dft_x = dft_x[..., 1]  # use only the positive frequencies
    else:
        raise ValueError("Unsupported DCT type")
    
    # Apply normalization
    if norm == 'ortho':
        scale = torch.sqrt(torch.tensor(2.0, dtype=x.dtype))  # Orthogonal normalization
    else:
        scale = 1.0
    
    return scale * dft_x


def gaussian(window_size, sigma):
    gauss = torch.Tensor([torch.exp(-(torch.tensor(x) - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return torch.tensor(window)

def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    C1 = 0.01**2
    C2 = 0.03**2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim_loss(img1, img2):
    return 1 - ssim(img1, img2)

class MelCepstralDistance(nn.Module):
    def __init__(self, n_coeffs):
        super(MelCepstralDistance, self).__init__()
        self.n_coeffs = n_coeffs
        self.constant = 10.0 / torch.log(torch.tensor(10.0)) * torch.sqrt(torch.tensor(2.0))
    
    def forward(self, mel1, mel2):
        """
        Compute the Mel Cepstral Distance between two MFCC sequences.
        
        Args:
        mfcc1 (torch.Tensor): The first MFCC sequence with shape (batch_size, seq_length, n_coeffs).
        mfcc2 (torch.Tensor): The second MFCC sequence with shape (batch_size, seq_length, n_coeffs).
        
        Returns:
        torch.Tensor: The MCD loss.
        """
        mfcc1 = dct1(mel1)[:,:self.n_coeffs]
        #print(mfcc1.shape)
        mfcc2 = dct1(mel2)[:,:self.n_coeffs]
        #print(mfcc2.shape)
        assert mfcc1.shape == mfcc2.shape, "Input tensors must have the same shape."
        assert mfcc1.shape[1] == self.n_coeffs, "The number of coefficients must match the initialized value."
        
        # Compute the squared difference between the MFCC coefficients
        diff = mfcc1 - mfcc2
        diff_squared = diff ** 2
        
        # Sum over the coefficient dimension
        diff_squared_sum = torch.sum(diff_squared, dim=1)
        
        # Compute the mean over the sequence length dimension
        diff_squared_sum_mean = torch.mean(diff_squared_sum, dim=0)
        
        # Compute the final MCD value
        mcd = self.constant * torch.sqrt(torch.abs(diff_squared_sum_mean))
        
        return torch.mean(mcd)
    

class SpectralConvergenceLoss(torch.nn.Module):
    def __init__(self):
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, predicted, target):
        numerator = torch.norm(predicted - target, p='fro')
        denominator = torch.norm(target, p='fro')
        return numerator / (denominator + 1e-9)


class GLU(nn.Module):
  def __init__(self):
    super(GLU, self).__init__()

  def forward(self, x):
    return x*torch.sigmoid(x)


class HypSnake(nn.Module):#learnable a
  def __init__(self,shape):
      super(HypSnake, self).__init__()
      self.a = nn.Parameter(torch.zeros(shape))
      self.first = True
  def forward(self, x):
    if self.first:
        self.first = False
        a = torch.zeros_like(x[0]).normal_(mean=0,std=50).abs()
        self.a = nn.Parameter(a)
    return torch.tanh(x + (torch.sin(self.a * x) ** 2) / self.a)


class MSDLoss(nn.Module):
    def __init__(self, n_fft=256, hop_length=128):
        super(MSDLoss, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, x, y):
        # x: (batch_size, num_mel_bins, num_frames)
        # y: (batch_size, num_mel_bins, num_frames)
        if x.shape != y.shape:
            raise ValueError("Input tensors must have the same shape")

        # Compute the STFT to obtain modulation spectra
        X = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)
        Y = torch.stft(y, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)
        
        # Compute the magnitude spectra
        X_mag = torch.abs(X)
        Y_mag = torch.abs(Y)

        # Calculate the mean squared deviation between the modulation spectra
        msd_loss = torch.mean((X_mag - Y_mag) ** 2)
        return msd_loss
    
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, mel_spectrogram1, mel_spectrogram2):
        # Add channel dimension
        mel_spectrogram1 = mel_spectrogram1.unsqueeze(1)
        mel_spectrogram2 = mel_spectrogram2.unsqueeze(1)
        return ssim_loss(mel_spectrogram1, mel_spectrogram2)

class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=[1024, 2048, 1920,512], 
                 hop_sizes=[256, 512,480, 108], 
                 win_lengths=[1024, 2048,1920, 256]):
        super(MultiResolutionSTFTLoss, self).__init__()
        
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        
        self.sc_loss = SpectralConvergenceLoss()
        self.mag_loss = LogSTFTMagnitudeLoss()

    def forward(self, y_pred, y_true):
        """
        Returns:
            Tensor: Combined STFT loss (sc_loss + mag_loss)
        """
        total_sc_loss = 0.0
        total_mag_loss = 0.0
        
        for fft_size, hop_size, win_length in zip(self.fft_sizes, 
                                                 self.hop_sizes, 
                                                 self.win_lengths):
            pred_stft = torch.stft(y_pred.squeeze(1), 
                                 n_fft=fft_size, 
                                 hop_length=hop_size,
                                 win_length=win_length,
                                 window=torch.hann_window(win_length).to(y_pred.device),
                                 return_complex=True).cuda()

            true_stft = torch.stft(y_true.squeeze(1), 
                                  n_fft=fft_size, 
                                  hop_length=hop_size,
                                  win_length=win_length,
                                  window=torch.hann_window(win_length).to(y_true.device),
                                  return_complex=True).cuda()
            
            sc_loss = self.sc_loss(pred_stft.abs(), true_stft.abs())
            mag_loss = self.mag_loss(pred_stft.abs(), true_stft.abs())
            
            total_sc_loss += sc_loss
            total_mag_loss += mag_loss
        
        num_resolutions = len(self.fft_sizes)
        return (total_sc_loss + total_mag_loss) / num_resolutions

class SpectralConvergenceLoss(nn.Module):
    """Spectral Convergence Loss"""
    def __init__(self):
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, pred_mag, target_mag):
        """
        Args:
            pred_mag: Magnitude of predicted STFT
            target_mag: Magnitude of target STFT
        """
        return torch.norm(target_mag - pred_mag, p='fro') / torch.norm(target_mag, p='fro')

class LogSTFTMagnitudeLoss(nn.Module):
    """Log STFT Magnitude Loss"""
    def __init__(self):
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, pred_mag, target_mag):
        """
        Args:
            pred_mag: Magnitude of predicted STFT
            target_mag: Magnitude of target STFT
        """
        return F.l1_loss(torch.log(pred_mag + 1e-8), torch.log(target_mag + 1e-8))

class MelSpectrogramLoss(nn.Module):
    """
    Mel-Spectrogram Loss for comparing audio in the mel-frequency domain
    """
    def __init__(self, 
                 sample_rate=48000,
                 n_fft=1920,
                 hop_length=480,
                 n_mels=128,
                 normalize=True,
                 mel_scale='slaney',
                 win_length=1920,
                 norm = 'slaney'):
        super(MelSpectrogramLoss, self).__init__()
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            normalized=normalize,
            mel_scale=mel_scale,
            win_length=win_length,
            norm = norm
        ).cuda()
        
        # L1 loss for comparing mel spectrograms
        self.l1_loss = nn.L1Loss()
        
        # Log loss for better handling of small values
        self.log_loss = LogMelLoss()

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred (torch.Tensor): Predicted waveform
            y_true (torch.Tensor): Ground truth waveform
        """
        # Compute mel spectrograms
        pred_mel = self.mel_transform(y_pred)
        true_mel = self.mel_transform(y_true)
        
        # Compute both L1 and Log losses
        l1_loss = self.l1_loss(pred_mel, true_mel)
        log_loss = self.log_loss(pred_mel, true_mel)
        
        # Combine losses
        return l1_loss + log_loss

class LogMelLoss(nn.Module):
    """Log-scale Mel-spectrogram Loss"""
    def __init__(self):
        super(LogMelLoss, self).__init__()

    def forward(self, pred_mel, target_mel):
        """
        Args:
            pred_mel: Predicted mel spectrogram
            target_mel: Target mel spectrogram
        """
        return F.mse_loss(
            torch.log(pred_mel.squeeze(1) + 1e-8),
            torch.log(target_mel.squeeze(1) + 1e-8)
        )


def kl_divergence_loss(V, V_hat):
    return torch.sum(V * torch.log((V + 1e-10) / (V_hat + 1e-10)) - V + V_hat)

def mse_loss(V, V_hat):
    return nn.MSELoss()(V, V_hat)

def SpecLoss(m1,m2):
    m1=m1#.detach().cpu().numpy()
    m2=m2
    #sl=torch.abs(torch.log(torch.abs(torch.from_numpy(dct(m1))+0.0000001))-torch.log(torch.abs(torch.from_numpy(dct(m2))+0.0000001)))
    sl=torch.abs(dct1(m1)-dct1(m2))
    #sll=torch.from_numpy(sl)
    return sl

def reconstruct(H, W):
        pad_size = (W.shape[2] - 1, W.shape[3] - 1)
        out = torch.nn.functional.conv2d(H, W.flip((2, 3)), padding=pad_size)
        return out


def dct1(x, type=2, norm='ortho'):

    dct_func = {
        1: torch.fft.fft,
        2: torch.fft.rfft,
        3: torch.fft.rfft,
    }[type]
    
    # Compute DFT along the last dimension
    dft_x = dct_func(x, dim=-1)
    
    # Select appropriate components for the DCT
    if type == 1:
        dft_x = dft_x[..., 0]  # use only the real part
    elif type == 2:
        dft_x = dft_x[..., 1]  # use only the positive frequencies
    else:
        raise ValueError("Unsupported DCT type")
    
    # Apply normalization
    if norm == 'ortho':
        scale = torch.sqrt(torch.tensor(2.0, dtype=x.dtype))  # Orthogonal normalization
    else:
        scale = 1.0
    
    return scale * dft_x


def gaussian(window_size, sigma):
    gauss = torch.Tensor([torch.exp(-(torch.tensor(x) - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return torch.tensor(window)

def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    C1 = 0.01**2
    C2 = 0.03**2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim_loss(img1, img2):
    return 1 - ssim(img1, img2)

class MelCepstralDistance(nn.Module):
    def __init__(self, n_coeffs):
        super(MelCepstralDistance, self).__init__()
        self.n_coeffs = n_coeffs
        self.constant = 10.0 / torch.log(torch.tensor(10.0)) * torch.sqrt(torch.tensor(2.0))
    
    def forward(self, mel1, mel2):
        """
        Compute the Mel Cepstral Distance between two MFCC sequences.
        
        Args:
        mfcc1 (torch.Tensor): The first MFCC sequence with shape (batch_size, seq_length, n_coeffs).
        mfcc2 (torch.Tensor): The second MFCC sequence with shape (batch_size, seq_length, n_coeffs).
        
        Returns:
        torch.Tensor: The MCD loss.
        """
        mfcc1 = dct1(mel1)[:,:self.n_coeffs]
        #print(mfcc1.shape)
        mfcc2 = dct1(mel2)[:,:self.n_coeffs]
        #print(mfcc2.shape)
        assert mfcc1.shape == mfcc2.shape, "Input tensors must have the same shape."
        assert mfcc1.shape[1] == self.n_coeffs, "The number of coefficients must match the initialized value."
        
        # Compute the squared difference between the MFCC coefficients
        diff = mfcc1 - mfcc2
        diff_squared = diff ** 2
        
        # Sum over the coefficient dimension
        diff_squared_sum = torch.sum(diff_squared, dim=1)
        
        # Compute the mean over the sequence length dimension
        diff_squared_sum_mean = torch.mean(diff_squared_sum, dim=0)
        
        # Compute the final MCD value
        mcd = self.constant * torch.sqrt(torch.abs(diff_squared_sum_mean))
        
        return torch.mean(mcd)
    

class SpectralConvergenceLoss(torch.nn.Module):
    def __init__(self):
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, predicted, target):
        numerator = torch.norm(predicted - target, p='fro')
        denominator = torch.norm(target, p='fro')
        return numerator / (denominator + 1e-9)


class GLU(nn.Module):
  def __init__(self):
    super(GLU, self).__init__()

  def forward(self, x):
    return x*torch.sigmoid(x)


class HypSnake(nn.Module):#learnable a
  def __init__(self,shape):
      super(HypSnake, self).__init__()
      self.a = nn.Parameter(torch.zeros(shape))
      self.first = True
  def forward(self, x):
    if self.first:
        self.first = False
        a = torch.zeros_like(x[0]).normal_(mean=0,std=50).abs()
        self.a = nn.Parameter(a)
    return torch.tanh(x + (torch.sin(self.a * x) ** 2) / self.a)


class MSDLoss(nn.Module):
    def __init__(self, n_fft=256, hop_length=128):
        super(MSDLoss, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, x, y):
        # x: (batch_size, num_mel_bins, num_frames)
        # y: (batch_size, num_mel_bins, num_frames)
        if x.shape != y.shape:
            raise ValueError("Input tensors must have the same shape")

        # Compute the STFT to obtain modulation spectra
        X = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)
        Y = torch.stft(y, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)
        
        # Compute the magnitude spectra
        X_mag = torch.abs(X)
        Y_mag = torch.abs(Y)

        # Calculate the mean squared deviation between the modulation spectra
        msd_loss = torch.mean((X_mag - Y_mag) ** 2)
        return msd_loss
    
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, mel_spectrogram1, mel_spectrogram2):
        # Add channel dimension
        mel_spectrogram1 = mel_spectrogram1.unsqueeze(1)
        mel_spectrogram2 = mel_spectrogram2.unsqueeze(1)
        return ssim_loss(mel_spectrogram1, mel_spectrogram2)

class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=[1024, 2048, 1920,512], 
                 hop_sizes=[256, 512,480, 108], 
                 win_lengths=[1024, 2048,1920, 256]):
        super(MultiResolutionSTFTLoss, self).__init__()
        
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        
        self.sc_loss = SpectralConvergenceLoss()
        self.mag_loss = LogSTFTMagnitudeLoss()

    def forward(self, y_pred, y_true):
        """
        Returns:
            Tensor: Combined STFT loss (sc_loss + mag_loss)
        """
        total_sc_loss = 0.0
        total_mag_loss = 0.0
        
        for fft_size, hop_size, win_length in zip(self.fft_sizes, 
                                                 self.hop_sizes, 
                                                 self.win_lengths):
            pred_stft = torch.stft(y_pred.squeeze(1), 
                                 n_fft=fft_size, 
                                 hop_length=hop_size,
                                 win_length=win_length,
                                 window=torch.hann_window(win_length).to(y_pred.device),
                                 return_complex=True).cuda()

            true_stft = torch.stft(y_true.squeeze(1), 
                                  n_fft=fft_size, 
                                  hop_length=hop_size,
                                  win_length=win_length,
                                  window=torch.hann_window(win_length).to(y_true.device),
                                  return_complex=True).cuda()
            
            sc_loss = self.sc_loss(pred_stft.abs(), true_stft.abs())
            mag_loss = self.mag_loss(pred_stft.abs(), true_stft.abs())
            
            total_sc_loss += sc_loss
            total_mag_loss += mag_loss
        
        num_resolutions = len(self.fft_sizes)
        return (total_sc_loss + total_mag_loss) / num_resolutions

class SpectralConvergenceLoss(nn.Module):
    """Spectral Convergence Loss"""
    def __init__(self):
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, pred_mag, target_mag):
        """
        Args:
            pred_mag: Magnitude of predicted STFT
            target_mag: Magnitude of target STFT
        """
        return torch.norm(target_mag - pred_mag, p='fro') / torch.norm(target_mag, p='fro')

class LogSTFTMagnitudeLoss(nn.Module):
    """Log STFT Magnitude Loss"""
    def __init__(self):
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, pred_mag, target_mag):
        """
        Args:
            pred_mag: Magnitude of predicted STFT
            target_mag: Magnitude of target STFT
        """
        return F.l1_loss(torch.log(pred_mag + 1e-8), torch.log(target_mag + 1e-8))

class MelSpectrogramLoss(nn.Module):
    """
    Mel-Spectrogram Loss for comparing audio in the mel-frequency domain
    """
    def __init__(self, 
                 sample_rate=48000,
                 n_fft=1920,
                 hop_length=480,
                 n_mels=128,
                 normalize=True,
                 mel_scale='slaney',
                 win_length=1920,
                 norm = 'slaney'):
        super(MelSpectrogramLoss, self).__init__()
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            normalized=normalize,
            mel_scale=mel_scale,
            win_length=win_length,
            norm = norm
        ).cuda()
        
        # L1 loss for comparing mel spectrograms
        self.l1_loss = nn.L1Loss()
        
        # Log loss for better handling of small values
        self.log_loss = LogMelLoss()

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred (torch.Tensor): Predicted waveform
            y_true (torch.Tensor): Ground truth waveform
        """
        # Compute mel spectrograms
        pred_mel = self.mel_transform(y_pred)
        true_mel = self.mel_transform(y_true)
        
        # Compute both L1 and Log losses
        l1_loss = self.l1_loss(pred_mel, true_mel)
        log_loss = self.log_loss(pred_mel, true_mel)
        
        # Combine losses
        return l1_loss + log_loss

class LogMelLoss(nn.Module):
    """Log-scale Mel-spectrogram Loss"""
    def __init__(self):
        super(LogMelLoss, self).__init__()

    def forward(self, pred_mel, target_mel):
        """
        Args:
            pred_mel: Predicted mel spectrogram
            target_mel: Target mel spectrogram
        """
        return F.mse_loss(
            torch.log(pred_mel.squeeze(1) + 1e-8),
            torch.log(target_mel.squeeze(1) + 1e-8)
        )

class CombinedVocoderLoss(nn.Module):
    """
    Combined loss function for vocoder training, incorporating multiple loss terms
    """
    def __init__(self, 
                 sample_rate=16000,
                 lambda_stft=1.0,
                 lambda_mel=1.0,
                 lambda_l1=1.0):
        super(CombinedVocoderLoss, self).__init__()
        
        self.stft_loss = MultiResolutionSTFTLoss()
        self.mel_loss = MelSpectrogramLoss(sample_rate=sample_rate)
        self.l1_loss = nn.L1Loss()
        
        self.lambda_stft = lambda_stft
        self.lambda_mel = lambda_mel
        self.lambda_l1 = lambda_l1

    def forward(self, y_pred, y_true):
        # Compute STFT loss (now returns a single tensor)
        stft_loss = self.stft_loss(y_pred, y_true)
        
        # Compute mel-spectrogram loss
        mel_loss = self.mel_loss(y_pred, y_true)
        
        # Compute time-domain L1 loss
        l1_loss = self.l1_loss(y_pred, y_true)
        
        # Combine all losses
        total_loss = (self.lambda_stft * stft_loss + 
                     self.lambda_mel * mel_loss + 
                     self.lambda_l1 * l1_loss)
        
        # Return both total loss and individual components
        loss_dict = {
            'total_loss': total_loss,
            'stft_loss': stft_loss,
            'mel_loss': mel_loss,
            'l1_loss': l1_loss
        }
        
        return total_loss, loss_dict

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, real_feats, fake_feats):
        losses = 0
        for r_feat, f_feat in zip(real_feats, fake_feats):
            losses += self.l1_loss(f_feat, r_feat.detach())
        return F.sigmoid(losses / len(real_feats))

class RelativeDiscriminator(nn.Module):
    def __init__(self):
        super(RelativeDiscriminator, self).__init__()
        
        # Feature extraction layers
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=15, stride=1, padding=7),
                nn.LeakyReLU(0.2),
                nn.Conv1d(32, 64, kernel_size=41, stride=4, groups=4, padding=20),
                nn.LeakyReLU(0.2),
                nn.Conv1d(64, 128, kernel_size=41, stride=4, groups=16, padding=20),
                nn.LeakyReLU(0.2),
                nn.Conv1d(128, 256, kernel_size=41, stride=4, groups=16, padding=20),
                nn.LeakyReLU(0.2),
            )
        ])
        
        # Final classification layer
        self.fc = nn.Linear(256, 1)
    
    def forward(self, x):
        # Feature extraction
        fmap = []
        for layer in self.conv_layers:
            x = layer(x)
            fmap.append(x)
        
        # Global average pooling
        x = torch.mean(x, dim=2)
        
        # Classification
        x = self.fc(x)
        return x
    

def split_subbands(waveform, cutoff_low=1000, cutoff_mid=8000, sample_rate=48000):
    """
    Splits the full-band waveform into low, mid, and high sub-bands.
    Example using simple filter design from torchaudio functional:
    In practice, you might use better filter designs or learned filters.
    """
    # Design simple filters (lowpass, bandpass, highpass)
    # For simplicity, we’ll do a lowpass at cutoff_low, 
    # then a highpass at cutoff_low and lowpass again at cutoff_mid, 
    # and a highpass at cutoff_mid for the high band.
    # In reality, consider more sophisticated filter design or a filterbank.
    
    lowpass_filter = torchaudio.functional.lowpass_biquad(waveform, sample_rate=sample_rate, cutoff_freq=cutoff_low)
    highpass_low = torchaudio.functional.highpass_biquad(waveform, sample_rate=sample_rate, cutoff_freq=cutoff_low)
    mid_band = torchaudio.functional.lowpass_biquad(highpass_low, sample_rate=sample_rate, cutoff_freq=cutoff_mid)
    high_band = torchaudio.functional.highpass_biquad(waveform, sample_rate=sample_rate, cutoff_freq=cutoff_mid)

    return lowpass_filter, mid_band, high_band


def compute_stft(x, n_fft, hop_length, win_length):
    window = torch.hann_window(win_length, device=x.device)
    return torch.stft(
        x, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, return_complex=True
    )

def spectral_angle_loss(pred_mel, target_mel, eps=1e-8):
    """
    pred_mel: (B, F, T) predicted mel spectrogram
    target_mel: (B, F, T) target mel spectrogram
    We assume mel spectrograms are non-negative. If needed, apply normalization beforehand.
    """
    # Flatten or treat each time-frequency frame as a vector
    # Here we'll compute similarity across frequency bins at each time frame
    # Adjust reduction as needed
    # shape: (B, F, T) -> (B*T, F)
    B, F, T = pred_mel.size()
    pred = pred_mel.permute(0,2,1).reshape(B*T, F)
    target = target_mel.permute(0,2,1).reshape(B*T, F)

    # Normalize vectors to unit length
    pred_norm = pred / (pred.norm(dim=1, keepdim=True) + eps)
    target_norm = target / (target.norm(dim=1, keepdim=True) + eps)

    # Compute cosine similarity and then a loss (1 - similarity)
    cos_sim = (pred_norm * target_norm).sum(dim=1)
    loss = 1.0 - cos_sim.mean()

    return loss

def modulation_spectral_distortion_loss(mel_pred, mel_real, eps=1e-8):
    """
    Compute the Modulation Spectral Distortion (MSD) between two mel spectrograms.
    
    Args:
        mel_pred: Predicted mel spectrogram (B, F, T)
        mel_real: Reference mel spectrogram (B, F, T)
        eps: Small epsilon for numerical stability.
        
    Returns:
        A scalar tensor representing the MSD loss.
    """
    # Ensure both have the same shape
    B, F, T = mel_pred.shape
    assert mel_real.shape == (B, F, T), "Mel spectrograms must have the same shape."

    # Compute modulation spectra along the time axis.
    # Use rFFT since mel spectrograms are real and we only need half the spectrum.
    pred_mod = torch.fft.rfft(mel_pred, dim=-1)  # (B, F, T_mod)
    real_mod = torch.fft.rfft(mel_real, dim=-1)  # (B, F, T_mod)
    
    # Extract magnitude
    pred_mod_mag = torch.abs(pred_mod) + eps
    real_mod_mag = torch.abs(real_mod) + eps

    # Convert to log domain
    pred_mod_log = torch.log(pred_mod_mag)
    real_mod_log = torch.log(real_mod_mag)

    # Compute L1 loss on log-magnitude modulation spectra
    msd_loss = torch.nn.functional.l1_loss(pred_mod_log, real_mod_log)

    return msd_loss


def earth_movers_distance_loss(pred, target, eps=1e-8):
    """
    Compute the Earth Mover's Distance (EMD) between predicted and target spectral distributions.
    
    Args:
        pred: Predicted spectrogram (B, F, T) - non-negative values representing magnitudes per frequency bin.
        target: Target spectrogram (B, F, T) - same shape as pred.
        eps: Small epsilon for numerical stability.
        
    Returns:
        A scalar tensor representing the averaged EMD across the batch and time frames.
    """
    B, F, T = pred.shape
    # Add epsilon to avoid division by zero
    pred_sum = pred.sum(dim=1, keepdim=True) + eps    # (B,1,T)
    target_sum = target.sum(dim=1, keepdim=True) + eps
    
    # Normalize to interpret as distributions
    pred_dist = pred / pred_sum    # (B,F,T)
    target_dist = target / target_sum
    
    # Compute cumulative distributions along frequency axis
    pred_cdf = pred_dist.cumsum(dim=1)   # (B,F,T)
    target_cdf = target_dist.cumsum(dim=1) # (B,F,T)
    
    # EMD per frame: sum of absolute differences of CDFs along frequency
    # EMD_frame(b,t) = sum over f (|pred_cdf(b,f,t) - target_cdf(b,f,t)|)
    emd_per_frame = torch.abs(pred_cdf - target_cdf).sum(dim=1)  # (B,T)
    
    # Average EMD over time and batch
    emd_loss = emd_per_frame.mean()
    return emd_loss

def phase_loss(fake_audio, real_audio, n_fft=1920, hop_length=480, win_length=1920, window_fn=torch.hann_window):
    """
    Compute a phase consistency loss between fake and real audio.
    This loss encourages the generated audio to match the phase structure of the real audio.
    """
    device = fake_audio.device
    window = window_fn(win_length).to(device)

    # STFT for real and fake audio
    real_stft = torch.stft(
        real_audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length, 
        window=window, return_complex=True
    )
    fake_stft = torch.stft(
        fake_audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length, 
        window=window, return_complex=True
    )

    real_r ,real_i = real_stft.real,real_stft.imag
    fake_r ,fake_i = fake_stft.real,fake_stft.imag

    real_mag = torch.sqrt(real_r**2 + real_i**2)
    fake_mag = torch.sqrt(fake_r**2 + fake_i**2)

    theta = torch.atan2(real_i, real_r)
    theta_hat = torch.atan2(fake_i, fake_r)
    dif_theta = 2 * real_mag * torch.sin((theta_hat - theta)/2)  #0 <= dif_thera <= 2*mag

        #cos_theta = y_real / (torch.sqrt(y_real**2 + y_imag**2)+1e-8)
        #sin_theta = y_imag / (torch.sqrt(y_real**2 + y_imag**2)+1e-8)
        #cos_theta_hat = y_real_hat / (torch.sqrt(y_real_hat**2 + y_imag_hat**2)+1e-8)
        #sin_theta_hat = y_imag_hat / (torch.sqrt(y_real_hat**2 + y_imag_hat**2)+1e-8)
        #cos_dif_theta = cos_theta * cos_theta_hat + sin_theta * sin_theta_hat
        #sin_half_dif_theta_squared = (1 - cos_dif_theta) / 2
    dif_mag = fake_mag - real_mag
    loss = torch.mean(dif_mag**2 + dif_theta**2)
   
    # Extract phase
    # real_phase = torch.angle(real_stft)   # (B, F, T)
    # fake_phase = torch.angle(fake_stft)   # (B, F, T)

    # # Compute phase difference and wrap it to (-pi, pi)
    # phase_diff = fake_phase - real_phase
    # # Wrap the phase difference
    # phase_diff_wrapped = (phase_diff + math.pi) % (2 * math.pi) - math.pi

    # Compute L1 loss on phase differences
    return loss #F.l1_loss(phase_diff_wrapped, torch.zeros_like(phase_diff_wrapped))



class TorchSTFT(torch.nn.Module):
    def __init__(self, filter_length=1920, hop_length=480, win_length=1920, window='hann', device="cuda"):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.from_numpy(get_window(window, win_length, fftbins=True).astype(np.float32))
        self.device = device

    def transform(self, input_data):
        forward_transform = torch.stft(
            input_data,
            self.filter_length, self.hop_length, self.win_length, window=self.window.to(input_data.device()),
            return_complex=True)

        return torch.abs(forward_transform), torch.angle(forward_transform)

    def inverse(self, magnitude, phase):
        magnitude, phase = magnitude.to(self.device), phase.to(self.device)
        inverse_transform = torch.istft(
            magnitude * torch.exp(phase * 1j),
            self.filter_length, self.hop_length, self.win_length, window=self.window.to(self.device))

        return inverse_transform.unsqueeze(-2)  # unsqueeze to stay consistent with conv_transpose1d implementation

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction

def stable_phase_magnitude_loss(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor, 
    n_fft: int = 1920, 
    hop_length: int = 480, 
    win_length: int = 1920,
    window: torch.Tensor = torch.hann_window, 
    eps: float = 1e-8
) -> torch.Tensor:
    window = window(win_length).to(y_pred.device)

    """
    Computes a combined magnitude–phase loss between the predicted and 
    target audio waveforms by comparing their STFT outputs.

    Args:
        y_pred (torch.Tensor): Predicted audio waveform of shape (batch, time).
        y_true (torch.Tensor): Ground-truth audio waveform of shape (batch, time).
        n_fft (int, optional): Size of FFT. Default is 1024.
        hop_length (int, optional): Hop (stride) size between STFT windows. Default is 256.
        window (torch.Tensor, optional): Window function for STFT. Default is None.
        eps (float, optional): Small constant to avoid division-by-zero. Default is 1e-8.

    Returns:
        torch.Tensor: Scalar loss value (magnitude loss + phase loss).
    """

    # 1) Compute STFT (returns complex tensors in PyTorch >= 1.7 if return_complex=True)
    #    For older PyTorch versions, it returns real/imag parts separately. Adjust accordingly.
    Y_pred = torch.stft(
        y_pred, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        win_length=win_length,
        window=window, 
        return_complex=True
    )  # (batch, freq, frames)
    
    Y_true = torch.stft(
        y_true, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        win_length=win_length,
        window=window, 
        return_complex=True
    )

    # 2) Magnitude loss
    #    Use safe sqrt (abs) to avoid NaNs or inf when magnitudes are zero or very small.
    pred_mag = torch.abs(Y_pred) + eps
    true_mag = torch.abs(Y_true) + eps
    
    # Example: L1 magnitude loss or MSE. You can choose whichever suits better.
    mag_loss = F.l1_loss(pred_mag, true_mag)
    # or, for MSE:
    # mag_loss = F.mse_loss(pred_mag, true_mag)

    # 3) Phase loss
    #    Option A: Directly compare imaginary parts (simple approach).
    #    Option B: Compare angles. PyTorch provides `torch.angle` for complex tensors.
    #
    #    Below we use Option A: MSE on imaginary parts.
    pred_imag = Y_pred.imag
    true_imag = Y_true.imag
    phase_loss = F.mse_loss(pred_imag, true_imag)
    
    # (Alternatively, you could do):
    #   pred_angle = torch.angle(Y_pred)
    #   true_angle = torch.angle(Y_true)
    #   phase_loss = F.l1_loss(pred_angle, true_angle)
    # or a cosine similarity approach, etc.

    # 4) Total loss
    total_loss = mag_loss + phase_loss

    return total_loss

class StableCosineDistanceLoss(nn.Module):
    """
    Computes a numerically stable cosine distance loss between x1 and x2.

    The cosine distance (1 - cosine similarity) is averaged over the batch.

    By clamping norms to a small epsilon, we avoid division-by-zero and ensure
    no NaNs in the forward pass.
    """
    def __init__(self, reduction='sum', eps=1e-8):
        """
        Args:
            reduction (str): Specifies the reduction to apply to the output.
                             Options are 'mean', 'sum', or 'none'.
            eps (float): Small value to avoid division by zero.
        """
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, x1, x2):
        """
        Args:
            x1 (torch.Tensor): Tensor of shape (N, D) or similar.
            x2 (torch.Tensor): Tensor of shape (N, D) or similar.

        Returns:
            torch.Tensor: The stable cosine distance loss.
        """
        # Normalize each vector by its L2 norm + eps
        x1_norm = x1 / (x1.norm(dim=1, keepdim=True).clamp(min=self.eps))
        x2_norm = x2 / (x2.norm(dim=1, keepdim=True).clamp(min=self.eps))

        # Compute cosine similarity
        cosine_sim = (x1_norm * x2_norm).sum(dim=1)

        # Convert similarity to distance
        cos_dist = 1.0 - cosine_sim

        # Reduce as requested
        if self.reduction == 'mean':
            return cos_dist.mean()
        elif self.reduction == 'sum':
            return cos_dist.sum()
        else:
            return cos_dist

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, real_feats, fake_feats):
        losses = 0
        for r_feat, f_feat in zip(real_feats, fake_feats):
            losses += self.l1_loss(f_feat, r_feat.detach())
        return F.sigmoid(losses / len(real_feats))

class RelativeDiscriminator(nn.Module):
    def __init__(self):
        super(RelativeDiscriminator, self).__init__()
        
        # Feature extraction layers
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=15, stride=1, padding=7),
                nn.LeakyReLU(0.2),
                nn.Conv1d(32, 64, kernel_size=41, stride=4, groups=4, padding=20),
                nn.LeakyReLU(0.2),
                nn.Conv1d(64, 128, kernel_size=41, stride=4, groups=16, padding=20),
                nn.LeakyReLU(0.2),
                nn.Conv1d(128, 256, kernel_size=41, stride=4, groups=16, padding=20),
                nn.LeakyReLU(0.2),
            )
        ])
        
        # Final classification layer
        self.fc = nn.Linear(256, 1)
    
    def forward(self, x):
        # Feature extraction
        fmap = []
        for layer in self.conv_layers:
            x = layer(x)
            fmap.append(x)
        
        # Global average pooling
        x = torch.mean(x, dim=2)
        
        # Classification
        x = self.fc(x)
        return x
    

def split_subbands(waveform, cutoff_low=1000, cutoff_mid=8000, sample_rate=48000):
    """
    Splits the full-band waveform into low, mid, and high sub-bands.
    Example using simple filter design from torchaudio functional:
    In practice, you might use better filter designs or learned filters.
    """
    # Design simple filters (lowpass, bandpass, highpass)
    # For simplicity, we’ll do a lowpass at cutoff_low, 
    # then a highpass at cutoff_low and lowpass again at cutoff_mid, 
    # and a highpass at cutoff_mid for the high band.
    # In reality, consider more sophisticated filter design or a filterbank.
    
    lowpass_filter = torchaudio.functional.lowpass_biquad(waveform, sample_rate=sample_rate, cutoff_freq=cutoff_low)
    highpass_low = torchaudio.functional.highpass_biquad(waveform, sample_rate=sample_rate, cutoff_freq=cutoff_low)
    mid_band = torchaudio.functional.lowpass_biquad(highpass_low, sample_rate=sample_rate, cutoff_freq=cutoff_mid)
    high_band = torchaudio.functional.highpass_biquad(waveform, sample_rate=sample_rate, cutoff_freq=cutoff_mid)

    return lowpass_filter, mid_band, high_band


def compute_stft(x, n_fft, hop_length, win_length):
    window = torch.hann_window(win_length, device=x.device)
    return torch.stft(
        x, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, return_complex=True
    )

def spectral_angle_loss(pred_mel, target_mel, eps=1e-8):
    """
    pred_mel: (B, F, T) predicted mel spectrogram
    target_mel: (B, F, T) target mel spectrogram
    We assume mel spectrograms are non-negative. If needed, apply normalization beforehand.
    """
    # Flatten or treat each time-frequency frame as a vector
    # Here we'll compute similarity across frequency bins at each time frame
    # Adjust reduction as needed
    # shape: (B, F, T) -> (B*T, F)
    B, F, T = pred_mel.size()
    pred = pred_mel.permute(0,2,1).reshape(B*T, F)
    target = target_mel.permute(0,2,1).reshape(B*T, F)

    # Normalize vectors to unit length
    pred_norm = pred / (pred.norm(dim=1, keepdim=True) + eps)
    target_norm = target / (target.norm(dim=1, keepdim=True) + eps)

    # Compute cosine similarity and then a loss (1 - similarity)
    cos_sim = (pred_norm * target_norm).sum(dim=1)
    loss = 1.0 - cos_sim.mean()

    return loss

def modulation_spectral_distortion_loss(mel_pred, mel_real, eps=1e-8):
    """
    Compute the Modulation Spectral Distortion (MSD) between two mel spectrograms.
    
    Args:
        mel_pred: Predicted mel spectrogram (B, F, T)
        mel_real: Reference mel spectrogram (B, F, T)
        eps: Small epsilon for numerical stability.
        
    Returns:
        A scalar tensor representing the MSD loss.
    """
    # Ensure both have the same shape
    B, F, T = mel_pred.shape
    assert mel_real.shape == (B, F, T), "Mel spectrograms must have the same shape."

    # Compute modulation spectra along the time axis.
    # Use rFFT since mel spectrograms are real and we only need half the spectrum.
    pred_mod = torch.fft.rfft(mel_pred, dim=-1)  # (B, F, T_mod)
    real_mod = torch.fft.rfft(mel_real, dim=-1)  # (B, F, T_mod)
    
    # Extract magnitude
    pred_mod_mag = torch.abs(pred_mod) + eps
    real_mod_mag = torch.abs(real_mod) + eps

    # Convert to log domain
    pred_mod_log = torch.log(pred_mod_mag)
    real_mod_log = torch.log(real_mod_mag)

    # Compute L1 loss on log-magnitude modulation spectra
    msd_loss = torch.nn.functional.l1_loss(pred_mod_log, real_mod_log)

    return msd_loss


def earth_movers_distance_loss(pred, target, eps=1e-8):
    """
    Compute the Earth Mover's Distance (EMD) between predicted and target spectral distributions.
    
    Args:
        pred: Predicted spectrogram (B, F, T) - non-negative values representing magnitudes per frequency bin.
        target: Target spectrogram (B, F, T) - same shape as pred.
        eps: Small epsilon for numerical stability.
        
    Returns:
        A scalar tensor representing the averaged EMD across the batch and time frames.
    """
    B, F, T = pred.shape
    # Add epsilon to avoid division by zero
    pred_sum = pred.sum(dim=1, keepdim=True) + eps    # (B,1,T)
    target_sum = target.sum(dim=1, keepdim=True) + eps
    
    # Normalize to interpret as distributions
    pred_dist = pred / pred_sum    # (B,F,T)
    target_dist = target / target_sum
    
    # Compute cumulative distributions along frequency axis
    pred_cdf = pred_dist.cumsum(dim=1)   # (B,F,T)
    target_cdf = target_dist.cumsum(dim=1) # (B,F,T)
    
    # EMD per frame: sum of absolute differences of CDFs along frequency
    # EMD_frame(b,t) = sum over f (|pred_cdf(b,f,t) - target_cdf(b,f,t)|)
    emd_per_frame = torch.abs(pred_cdf - target_cdf).sum(dim=1)  # (B,T)
    
    # Average EMD over time and batch
    emd_loss = emd_per_frame.mean()
    return emd_loss

def phase_loss(fake_audio, real_audio, n_fft=1920, hop_length=480, win_length=1920, window_fn=torch.hann_window):
    """
    Compute a phase consistency loss between fake and real audio.
    This loss encourages the generated audio to match the phase structure of the real audio.
    """
    device = fake_audio.device
    window = window_fn(win_length).to(device)

    # STFT for real and fake audio
    real_stft = torch.stft(
        real_audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length, 
        window=window, return_complex=True
    )
    fake_stft = torch.stft(
        fake_audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length, 
        window=window, return_complex=True
    )

    real_r ,real_i = real_stft.real,real_stft.imag
    fake_r ,fake_i = fake_stft.real,fake_stft.imag

    real_mag = torch.sqrt(real_r**2 + real_i**2)
    fake_mag = torch.sqrt(fake_r**2 + fake_i**2)

    theta = torch.atan2(real_i, real_r)
    theta_hat = torch.atan2(fake_i, fake_r)
    dif_theta = 2 * real_mag * torch.sin((theta_hat - theta)/2)  #0 <= dif_thera <= 2*mag

        #cos_theta = y_real / (torch.sqrt(y_real**2 + y_imag**2)+1e-8)
        #sin_theta = y_imag / (torch.sqrt(y_real**2 + y_imag**2)+1e-8)
        #cos_theta_hat = y_real_hat / (torch.sqrt(y_real_hat**2 + y_imag_hat**2)+1e-8)
        #sin_theta_hat = y_imag_hat / (torch.sqrt(y_real_hat**2 + y_imag_hat**2)+1e-8)
        #cos_dif_theta = cos_theta * cos_theta_hat + sin_theta * sin_theta_hat
        #sin_half_dif_theta_squared = (1 - cos_dif_theta) / 2
    dif_mag = fake_mag - real_mag
    loss = torch.mean(dif_mag**2 + dif_theta**2)
   
    # Extract phase
    # real_phase = torch.angle(real_stft)   # (B, F, T)
    # fake_phase = torch.angle(fake_stft)   # (B, F, T)

    # # Compute phase difference and wrap it to (-pi, pi)
    # phase_diff = fake_phase - real_phase
    # # Wrap the phase difference
    # phase_diff_wrapped = (phase_diff + math.pi) % (2 * math.pi) - math.pi

    # Compute L1 loss on phase differences
    return loss #F.l1_loss(phase_diff_wrapped, torch.zeros_like(phase_diff_wrapped))



class TorchSTFT(torch.nn.Module):
    def __init__(self, filter_length=1920, hop_length=480, win_length=1920, window='hann', device="cuda"):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.from_numpy(get_window(window, win_length, fftbins=True).astype(np.float32))
        self.device = device

    def transform(self, input_data):
        forward_transform = torch.stft(
            input_data,
            self.filter_length, self.hop_length, self.win_length, window=self.window.to(input_data.device()),
            return_complex=True)

        return torch.abs(forward_transform), torch.angle(forward_transform)

    def inverse(self, magnitude, phase):
        magnitude, phase = magnitude.to(self.device), phase.to(self.device)
        inverse_transform = torch.istft(
            magnitude * torch.exp(phase * 1j),
            self.filter_length, self.hop_length, self.win_length, window=self.window.to(self.device))

        return inverse_transform.unsqueeze(-2)  # unsqueeze to stay consistent with conv_transpose1d implementation

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction

def stable_phase_magnitude_loss(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor, 
    n_fft: int = 1920, 
    hop_length: int = 480, 
    win_length: int = 1920,
    window: torch.Tensor = torch.hann_window, 
    eps: float = 1e-8
) -> torch.Tensor:
    window = window(win_length).to(y_pred.device)

    """
    Computes a combined magnitude–phase loss between the predicted and 
    target audio waveforms by comparing their STFT outputs.

    Args:
        y_pred (torch.Tensor): Predicted audio waveform of shape (batch, time).
        y_true (torch.Tensor): Ground-truth audio waveform of shape (batch, time).
        n_fft (int, optional): Size of FFT. Default is 1024.
        hop_length (int, optional): Hop (stride) size between STFT windows. Default is 256.
        window (torch.Tensor, optional): Window function for STFT. Default is None.
        eps (float, optional): Small constant to avoid division-by-zero. Default is 1e-8.

    Returns:
        torch.Tensor: Scalar loss value (magnitude loss + phase loss).
    """

    # 1) Compute STFT (returns complex tensors in PyTorch >= 1.7 if return_complex=True)
    #    For older PyTorch versions, it returns real/imag parts separately. Adjust accordingly.
    Y_pred = torch.stft(
        y_pred, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        win_length=win_length,
        window=window, 
        return_complex=True
    )  # (batch, freq, frames)
    
    Y_true = torch.stft(
        y_true, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        win_length=win_length,
        window=window, 
        return_complex=True
    )

    # 2) Magnitude loss
    #    Use safe sqrt (abs) to avoid NaNs or inf when magnitudes are zero or very small.
    pred_mag = torch.abs(Y_pred) + eps
    true_mag = torch.abs(Y_true) + eps
    
    # Example: L1 magnitude loss or MSE. You can choose whichever suits better.
    mag_loss = F.l1_loss(pred_mag, true_mag)
    # or, for MSE:
    # mag_loss = F.mse_loss(pred_mag, true_mag)

    # 3) Phase loss
    #    Option A: Directly compare imaginary parts (simple approach).
    #    Option B: Compare angles. PyTorch provides `torch.angle` for complex tensors.
    #
    #    Below we use Option A: MSE on imaginary parts.
    pred_imag = Y_pred.imag
    true_imag = Y_true.imag
    phase_loss = F.mse_loss(pred_imag, true_imag)
    
    # (Alternatively, you could do):
    #   pred_angle = torch.angle(Y_pred)
    #   true_angle = torch.angle(Y_true)
    #   phase_loss = F.l1_loss(pred_angle, true_angle)
    # or a cosine similarity approach, etc.

    # 4) Total loss
    total_loss = mag_loss + phase_loss

    return total_loss

class StableCosineDistanceLoss(nn.Module):
    """
    Computes a numerically stable cosine distance loss between x1 and x2.

    The cosine distance (1 - cosine similarity) is averaged over the batch.

    By clamping norms to a small epsilon, we avoid division-by-zero and ensure
    no NaNs in the forward pass.
    """
    def __init__(self, reduction='sum', eps=1e-8):
        """
        Args:
            reduction (str): Specifies the reduction to apply to the output.
                             Options are 'mean', 'sum', or 'none'.
            eps (float): Small value to avoid division by zero.
        """
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, x1, x2):
        """
        Args:
            x1 (torch.Tensor): Tensor of shape (N, D) or similar.
            x2 (torch.Tensor): Tensor of shape (N, D) or similar.

        Returns:
            torch.Tensor: The stable cosine distance loss.
        """
        # Normalize each vector by its L2 norm + eps
        x1_norm = x1 / (x1.norm(dim=1, keepdim=True).clamp(min=self.eps))
        x2_norm = x2 / (x2.norm(dim=1, keepdim=True).clamp(min=self.eps))

        # Compute cosine similarity
        cosine_sim = (x1_norm * x2_norm).sum(dim=1)

        # Convert similarity to distance
        cos_dist = 1.0 - cosine_sim

        # Reduce as requested
        if self.reduction == 'mean':
            return cos_dist.mean()
        elif self.reduction == 'sum':
            return cos_dist.sum()
        else:
            return cos_dist

