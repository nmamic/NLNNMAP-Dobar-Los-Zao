import math
import torch
import torchaudio

from torch.utils.data import Dataset, DataLoader, random_split


NOTE_FREQS = torch.tensor([
    261.63,  # C4
    277.18,  # C#4 / Db4
    293.66,  # D4
    311.13,  # D#4 / Eb4
    329.63,  # E4
    349.23,  # F4
    369.99,  # F#4 / Gb4
    392.00,  # G4
    415.30,  # G#4 / Ab4
    440.00,  # A4
    466.16,  # A#4 / Bb4
    493.88,  # B4

    523.25,  # C5
    554.37,  # C#5 / Db5
    587.33,  # D5
    622.25,  # D#5 / Eb5
    659.25,  # E5
    698.46,  # F5
    739.99,  # F#5 / Gb5
    783.99,  # G5
    830.61,  # G#5 / Ab5
    880.00,  # A5
    932.33,  # A#5 / Bb5
    987.77,  # B5

    1046.50, # C6
])


CHORDS = {
    # Major triads
    "C_major":  [261.63, 329.63, 392.00],
    "Cs_major": [277.18, 349.23, 415.30],
    "D_major":  [293.66, 369.99, 440.00],
    "Ds_major": [311.13, 392.00, 466.16],
    "E_major":  [329.63, 415.30, 493.88],
    "F_major":  [349.23, 440.00, 523.25],
    "Fs_major": [369.99, 466.16, 554.37],
    "G_major":  [392.00, 493.88, 587.33],
    "Gs_major": [415.30, 523.25, 622.25],
    "A_major":  [440.00, 554.37, 659.25],
    "As_major": [466.16, 587.33, 698.46],
    "B_major":  [493.88, 622.25, 739.99],

    # Minor triads
    "C_minor":  [261.63, 311.13, 392.00],
    "Cs_minor": [277.18, 329.63, 415.30],
    "D_minor":  [293.66, 349.23, 440.00],
    "Ds_minor": [311.13, 369.99, 466.16],
    "E_minor":  [329.63, 392.00, 493.88],
    "F_minor":  [349.23, 415.30, 523.25],
    "Fs_minor": [369.99, 440.00, 554.37],
    "G_minor":  [392.00, 466.16, 587.33],
    "Gs_minor": [415.30, 493.88, 622.25],
    "A_minor":  [440.00, 523.25, 659.25],
    "As_minor": [466.16, 554.37, 698.46],
    "B_minor":  [493.88, 587.33, 739.99],

    # Diminished triads
    "C_dim":  [261.63, 311.13, 369.99],
    "D_dim":  [293.66, 349.23, 415.30],
    "E_dim":  [329.63, 392.00, 466.16],
    "F_dim":  [349.23, 415.30, 493.88],
    "G_dim":  [392.00, 466.16, 554.37],
    "A_dim":  [440.00, 523.25, 622.25],
    "B_dim":  [493.88, 587.33, 698.46],

    # Augmented triads
    "C_aug":  [261.63, 329.63, 415.30],
    "D_aug":  [293.66, 369.99, 466.16],
    "E_aug":  [329.63, 415.30, 523.25],
    "F_aug":  [349.23, 440.00, 554.37],
    "G_aug":  [392.00, 493.88, 622.25],
    "A_aug":  [440.00, 554.37, 698.46],
    "B_aug":  [493.88, 622.25, 783.99],

    # Suspended chords
    "C_sus2": [261.63, 293.66, 392.00],
    "C_sus4": [261.63, 349.23, 392.00],
    "D_sus2": [293.66, 329.63, 440.00],
    "D_sus4": [293.66, 392.00, 440.00],
    "E_sus2": [329.63, 369.99, 493.88],
    "E_sus4": [329.63, 440.00, 493.88],
    "F_sus2": [349.23, 392.00, 523.25],
    "F_sus4": [349.23, 466.16, 523.25],
    "G_sus2": [392.00, 440.00, 587.33],
    "G_sus4": [392.00, 523.25, 587.33],
    "A_sus2": [440.00, 493.88, 659.25],
    "A_sus4": [440.00, 587.33, 659.25],

    # Seventh chords
    "C_maj7": [261.63, 329.63, 392.00, 493.88],
    "C_min7": [261.63, 311.13, 392.00, 466.16],
    "C_7":    [261.63, 329.63, 392.00, 466.16],

    "D_maj7": [293.66, 369.99, 440.00, 554.37],
    "D_min7": [293.66, 349.23, 440.00, 523.25],
    "D_7":    [293.66, 369.99, 440.00, 523.25],

    "E_maj7": [329.63, 415.30, 493.88, 622.25],
    "E_min7": [329.63, 392.00, 493.88, 587.33],
    "E_7":    [329.63, 415.30, 493.88, 587.33],

    "F_maj7": [349.23, 440.00, 523.25, 659.25],
    "F_min7": [349.23, 415.30, 523.25, 622.25],
    "F_7":    [349.23, 440.00, 523.25, 622.25],

    "G_maj7": [392.00, 493.88, 587.33, 739.99],
    "G_min7": [392.00, 466.16, 587.33, 698.46],
    "G_7":    [392.00, 493.88, 587.33, 698.46],

    "A_maj7": [440.00, 554.37, 659.25, 830.61],
    "A_min7": [440.00, 523.25, 659.25, 783.99],
    "A_7":    [440.00, 554.37, 659.25, 783.99],
}


def make_clean_waveform(
    sample_rate=16000,
    num_samples=16256,
    regime="single_note",
):
    waveform = torch.zeros(num_samples)

    if regime == "single_note":
        freq = NOTE_FREQS[torch.randint(0, len(NOTE_FREQS), size=(1,)).item()].item()

        t = torch.arange(num_samples).float() / sample_rate
        amp = torch.empty(1).uniform_(0.3, 0.8).item()
        phase = torch.empty(1).uniform_(0.0, 2.0 * math.pi).item()

        waveform = amp * torch.sin(2.0 * math.pi * freq * t + phase)

        fade_samples = int(0.03 * sample_rate)
        fade_in = torch.linspace(0.0, 1.0, fade_samples)
        fade_out = torch.linspace(1.0, 0.0, fade_samples)

        waveform[:fade_samples] *= fade_in
        waveform[-fade_samples:] *= fade_out

    elif regime == "multi_note":
        # Multiple notes in sequence, not at the same time.
        num_notes = torch.randint(2, 6, size=(1,)).item()

        segment_length = num_samples // num_notes

        for note_idx in range(num_notes):
            start = note_idx * segment_length

            if note_idx == num_notes - 1:
                end = num_samples
            else:
                end = (note_idx + 1) * segment_length

            current_length = end - start

            freq = NOTE_FREQS[
                torch.randint(0, len(NOTE_FREQS), size=(1,)).item()
            ].item()

            t = torch.arange(current_length).float() / sample_rate

            amp = torch.empty(1).uniform_(0.3, 0.8).item()
            phase = torch.empty(1).uniform_(0.0, 2.0 * math.pi).item()

            note_waveform = amp * torch.sin(
                2.0 * math.pi * freq * t + phase
            )

            # Short fade per note to avoid clicks between notes.
            fade_samples = int(0.01 * sample_rate)
            fade_samples = min(fade_samples, current_length // 2)

            if fade_samples > 0:
                fade_in = torch.linspace(0.0, 1.0, fade_samples)
                fade_out = torch.linspace(1.0, 0.0, fade_samples)

                note_waveform[:fade_samples] *= fade_in
                note_waveform[-fade_samples:] *= fade_out

            waveform[start:end] = note_waveform

    elif regime == "chord":
        chord_names = list(CHORDS.keys())
        chord_name = chord_names[
            torch.randint(0, len(chord_names), size=(1,)).item()
        ]

        freqs = CHORDS[chord_name]

        t = torch.arange(num_samples).float() / sample_rate

        for freq in freqs:
            amp = torch.empty(1).uniform_(0.3, 0.8).item()
            phase = torch.empty(1).uniform_(0.0, 2.0 * math.pi).item()

            waveform += amp * torch.sin(
                2.0 * math.pi * freq * t + phase
            )

        fade_samples = int(0.03 * sample_rate)
        fade_in = torch.linspace(0.0, 1.0, fade_samples)
        fade_out = torch.linspace(1.0, 0.0, fade_samples)

        waveform[:fade_samples] *= fade_in
        waveform[-fade_samples:] *= fade_out

    else:
        raise ValueError(
            f"Unknown regime: {regime}. "
            "Use 'single_note', 'multi_note', or 'chord'."
        )

    waveform = waveform / (waveform.abs().max() + 1e-8)

    return waveform.unsqueeze(0)


def add_noise_at_snr(clean_waveform, snr_db=10.0):
    noise = torch.randn_like(clean_waveform)

    signal_power = clean_waveform.pow(2).mean()
    noise_power = noise.pow(2).mean()

    target_noise_power = signal_power / (10 ** (snr_db / 10))

    scale = torch.sqrt(target_noise_power / (noise_power + 1e-8))
    noisy_waveform = clean_waveform + scale * noise

    return noisy_waveform


def generate_waveform_pairs(
    num_examples=500,
    sample_rate=16000,
    num_samples=16256,
    regime="multi_note",
    min_snr_db=15.0,
    max_snr_db=30.0,
):
    clean_waveforms = []
    noisy_waveforms = []

    for _ in range(num_examples):
        clean = make_clean_waveform(
            sample_rate=sample_rate,
            num_samples=num_samples,
            regime=regime,
        )

        snr_db = torch.empty(1).uniform_(min_snr_db, max_snr_db).item()
        noisy = add_noise_at_snr(clean, snr_db=snr_db)

        clean_waveforms.append(clean)
        noisy_waveforms.append(noisy)

    clean_waveforms = torch.stack(clean_waveforms)
    noisy_waveforms = torch.stack(noisy_waveforms)

    return clean_waveforms, noisy_waveforms


def waveforms_to_normalized_spectrograms(
    clean_waveforms,
    noisy_waveforms,
    n_fft=510,
    win_length=510,
    hop_length=128,
):
    spectrogram_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        power=2.0,
    )

    clean_specs = spectrogram_transform(clean_waveforms)
    noisy_specs = spectrogram_transform(noisy_waveforms)

    clean_specs = torch.log1p(clean_specs)
    noisy_specs = torch.log1p(noisy_specs)

    all_specs = torch.cat([clean_specs, noisy_specs], dim=0)

    spec_min = all_specs.min()
    spec_max = all_specs.max()

    clean_specs = (clean_specs - spec_min) / (spec_max - spec_min + 1e-8)
    noisy_specs = (noisy_specs - spec_min) / (spec_max - spec_min + 1e-8)

    return clean_specs, noisy_specs, spec_min, spec_max


def generate_spectrogram_dataset(
    num_examples=500,
    sample_rate=16000,
    num_samples=16256,
    regime="multi_note",
    min_snr_db=15.0,
    max_snr_db=30.0,
    n_fft=510,
    win_length=510,
    hop_length=128,
):
    clean_waveforms, noisy_waveforms = generate_waveform_pairs(
        num_examples=num_examples,
        sample_rate=sample_rate,
        num_samples=num_samples,
        regime=regime,
        min_snr_db=min_snr_db,
        max_snr_db=max_snr_db,
    )

    clean_specs, noisy_specs, spec_min, spec_max = waveforms_to_normalized_spectrograms(
        clean_waveforms=clean_waveforms,
        noisy_waveforms=noisy_waveforms,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
    )

    return {
        "clean_waveforms": clean_waveforms,
        "noisy_waveforms": noisy_waveforms,
        "clean_specs": clean_specs,
        "noisy_specs": noisy_specs,
        "spec_min": spec_min,
        "spec_max": spec_max,
    }


class SpectrogramPairDataset(Dataset):
    def __init__(self, noisy_specs, clean_specs):
        self.noisy_specs = noisy_specs
        self.clean_specs = clean_specs

    def __len__(self):
        return len(self.clean_specs)

    def __getitem__(self, idx):
        noisy = self.noisy_specs[idx]
        clean = self.clean_specs[idx]

        return noisy, clean


def make_dataloaders(
    noisy_specs,
    clean_specs,
    batch_size=32,
    train_fraction=0.8,
):
    dataset = SpectrogramPairDataset(noisy_specs, clean_specs)

    train_size = int(train_fraction * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, dataset