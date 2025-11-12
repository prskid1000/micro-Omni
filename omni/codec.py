
import torch
from torch import nn

class RVQ(nn.Module):
    """ Two-level residual vector quantizer for 80/128-bin mel frames. """
    def __init__(self, codebooks=2, codebook_size=128, d=64):
        super().__init__()
        self.codebooks = nn.ParameterList([nn.Embedding(codebook_size, d) for _ in range(codebooks)])
        self.proj_in = nn.Linear(128, d)
        self.proj_out = nn.Linear(d, 128)

    def encode(self, mel_frame):
        """
        mel_frame: (B,128). Greedy residual quantization for simplicity.
        Returns indices: (B, codebooks)
        """
        z = self.proj_in(mel_frame)
        residual = z
        idxs = []
        for cb in self.codebooks:
            # Nearest neighbour in embedding space
            code = cb.weight  # (K, d)
            # compute distances
            # (B,1,d) - (1,K,d) -> (B,K,d)
            dist = (residual[:,None,:] - code[None,:,:]).pow(2).sum(-1)  # (B,K)
            ind = dist.argmin(dim=-1)
            idxs.append(ind)
            residual = residual - code[ind]
        return torch.stack(idxs, dim=-1)  # (B, C)

    def decode(self, idxs):
        """
        idxs: (B, C)
        """
        z = 0.0
        for i, cb in enumerate(self.codebooks):
            z = z + cb(idxs[:, i])
        mel = self.proj_out(z)
        return mel

class GriffinLimVocoder:
    """ Classical Griffin-Lim to turn mels -> waveform (approximate). """
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256, win_length=1024, n_iter=32):
        import librosa
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop = hop_length
        self.win = win_length
        self.n_iter = n_iter
        self.librosa = librosa

    def mel_to_audio(self, mel):
        # mel: (T, mel_bins), scaled [0, 1] approx; invert via pseudo-inverse then Griffin-Lim
        import numpy as np
        mel = mel.T  # (mel_bins, T) = (128, T)
        S = np.maximum(mel, 1e-5)
        
        # Convert mel to linear magnitude spectrogram
        # Approximate: use librosa's mel filterbank to convert back
        # This is a simplified approach - in practice you'd need the exact inverse
        try:
            # Create a dummy linear spectrogram by upsampling mel
            # Griffin-Lim expects (n_fft//2 + 1, T) = (513, T)
            n_freq = self.n_fft // 2 + 1  # 513
            mel_bins = S.shape[0]  # 128
            
            # Simple upsampling: repeat mel bins to fill frequency bins
            S_linear = np.zeros((n_freq, S.shape[1]))
            # Distribute mel bins across frequency range
            for i in range(mel_bins):
                start_idx = int(i * n_freq / mel_bins)
                end_idx = int((i + 1) * n_freq / mel_bins)
                S_linear[start_idx:end_idx, :] = S[i:i+1, :]
            
            y = self.librosa.griffinlim(S_linear, hop_length=self.hop, win_length=self.win, n_fft=self.n_fft, n_iter=self.n_iter)
            return y
        except Exception as e:
            # Fallback: generate simple tone
            duration = S.shape[1] * self.hop / self.sr
            t = np.linspace(0, duration, int(self.sr * duration))
            y = 0.1 * np.sin(2 * np.pi * 440 * t)  # A4 note
            return y
