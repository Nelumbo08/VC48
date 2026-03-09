import os
import torch
from torch.utils.data import DataLoader
from dataset import WaveformDataset
from models import *
from layers import *
from utils import *
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import datetime
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

# A = source, B = target ...... Adesh,Avani

vocoder = load_vocoder()
istft = TorchSTFT()

def train_step(config,dataloader_A,dataloader_B):
    gen = Generator().to(device)
    disc = Discriminator().to(device)

    optimizer_G = Adam(gen.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_D = Adam(disc.parameters(), lr=0.0001, betas=(0.5, 0.999))

    # dataloader_A = DataLoader(WaveformDataset(config['A_dir']), batch_size=config['batch_size'], shuffle=True)
    # dataloader_B = DataLoader(WaveformDataset(config['B_dir']), batch_size=config['batch_size'], shuffle=True)

    cycle_loss = nn.L1Loss()
    identity_loss = nn.L1Loss()
    f0_loss = nn.MSELoss()
    stft_loss = MultiResolutionSTFTLoss()
    cos_loss = StableCosineDistanceLoss()
    divergence_loss = kl_divergence_loss
    # Initialize tensorboard writer
    writer = SummaryWriter(f'runs/experiment_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}')
    
    # Helper function to create matplotlib figure
    def plot_spectrogram(spec, title):
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(spec.cpu().detach().numpy(), aspect='auto', origin='lower')
        plt.colorbar(im, ax=ax)
        ax.set_title(title)
        plt.close()
        return fig
        
    def plot_waveform(wav, title):
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.plot(wav.cpu().detach().numpy())
        ax.set_title(title)
        plt.close()
        return fig

    bs = config['batch_size']
    #training loop
    epochs = 1000
    step = 0
    for epoch in range(epochs):
        for source,target in (zip(dataloader_A,dataloader_B)):
            source_mel = source['mel_spec'].to(device)
            target_mel = target['mel_spec'].to(device)
            source_wav = source['waveform'].to(device)
            target_wav = target['waveform'].to(device)
            
            optimizer_D.zero_grad()
            gen.eval()
            disc.train()

            d_real = disc(target_mel)
            f_mel_d,W_f_d = gen(source_mel.unsqueeze(1))

            d_fake = disc(f_mel_d.detach())

            # Hinge loss for discriminator
            d_loss_real = torch.mean(torch.relu(1 - d_real))
            d_loss_fake = torch.mean(torch.relu(1 + d_fake))
            d_loss = (d_loss_real + d_loss_fake) / 2

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            gen.train()
            disc.eval()

            f_mel_g,W_f_g = gen(source_mel.unsqueeze(1))
            d_fake_g = disc(f_mel_g)

            f_mel_i,W_f_i = gen(f_mel_d.unsqueeze(1))
            d_fake_i = disc(f_mel_i)

            f_mel_b,W_f_b = gen(target_mel.unsqueeze(1))
            d_fake_b = disc(f_mel_b)

            # Hinge loss for generator
            g_adv_loss = -torch.mean(d_fake_g) - torch.mean(d_fake_i) - torch.mean(d_fake_b)

            fake_mag,fake_phase = vocoder(f_mel_g)
            real_mag,real_phase = vocoder(target_mel)
            
            fake_wav = istft.inverse(fake_mag,fake_phase)
            real_wav = istft.inverse(real_mag,real_phase)
            
            cyc_loss = cycle_loss(f_mel_i,f_mel_b)
            id_loss = identity_loss(f_mel_b,target_mel)

            f_loss = f0_loss(extract_f0(fake_wav),extract_f0(target_wav))
            stft_l = stft_loss(fake_wav,target_wav)
            cos_l = cos_loss(fake_wav,target_wav)

            div_loss = divergence_loss(W_f_b,W_f_g)

            # Combine losses
            g_loss = g_adv_loss + cyc_loss + id_loss #+ 0.2*f_loss + 0.1*stft_l + 0.01*cos_l + 0.01*div_loss

            g_loss.backward()
            optimizer_G.step()

            # Log every 100 steps
            if step % 100 == 0:
                # Log losses
                writer.add_scalar('Loss/discriminator', d_loss.item(), step)
                writer.add_scalar('Loss/generator', g_loss.item(), step)
                writer.add_scalar('Loss/cycle', cyc_loss.item(), step)
                writer.add_scalar('Loss/identity', id_loss.item(), step)
                # writer.add_scalar('Loss/f0', f_loss.item(), step)
                # writer.add_scalar('Loss/stft', stft_l.item(), step)
                # writer.add_scalar('Loss/cosine', cos_l.item(), step)
                # writer.add_scalar('Loss/divergence', div_loss.item(), step)
                # Log spectrograms
                writer.add_figure('Spectrograms/Real', 
                    plot_spectrogram(source_mel[0], 'Real Mel-Spectrogram'), step)
                writer.add_figure('Spectrograms/Fake', 
                    plot_spectrogram(f_mel_g[0], 'Generated Mel-Spectrogram'), step)
                writer.add_figure('Spectrograms/Target', 
                    plot_spectrogram(target_mel[0], 'Target Mel-Spectrogram'), step)
                
                # Log waveforms
                writer.add_figure('Waveforms/Real',
                    plot_waveform(source_wav[0].squeeze(), 'Real Waveform'), step)
                writer.add_figure('Waveforms/Fake',
                    plot_waveform(fake_wav[0].squeeze(), 'Generated Waveform'), step)
                writer.add_figure('Waveforms/Target',
                    plot_waveform(target_wav[0].squeeze(), 'Target Waveform'), step)
                
            step += 1
        print(f"Epoch {epoch+1}: G_loss- {g_loss.item()}  D_loss- {d_loss.item()}  f0_loss- {f_loss.item()}  stft_loss- {stft_l.item()}  cos_loss- {cos_l.item()}  div_loss- {div_loss.item()}")
    
    writer.close()


if __name__ == "__main__":
    config = {
    'A_dir': '/home/goquest/VC48/Data/AVANI',
    'B_dir': '/home/goquest/VC48/Data/ADESH',
    'batch_size': 32
    }    
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = '/home/goquest/VC48/checkpoints/vc1'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    dataset_A = WaveformDataset(config['A_dir'])
    dataset_B = WaveformDataset(config['B_dir'])
    dataloader_A = DataLoader(dataset_A, batch_size=config['batch_size'], shuffle=False, num_workers=4,drop_last=True)
    dataloader_B = DataLoader(dataset_B, batch_size=config['batch_size'], shuffle=False, num_workers=4,drop_last=True)

    print(len(dataloader_A))
    print(len(dataloader_B))
    
    # vocoder_gan = VocoderGAN(config)
    # vocoder_gan.train(loader)

    train_step(config,dataloader_A,dataloader_B)


#train_step(config)

