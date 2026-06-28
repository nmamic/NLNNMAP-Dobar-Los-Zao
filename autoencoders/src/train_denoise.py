import torch

from dataset_generation import (
    generate_waveform_pairs,
    waveforms_to_normalized_spectrograms,
    make_dataloaders,
)

from src.denoise_spec import DenoiseSpecAutoencoder

SEED = 123

SAMPLE_RATE = 16000
NUM_SAMPLES = 16256

N_FFT = 510
WIN_LENGTH = 510
HOP_LENGTH = 128

NUM_EXAMPLES_PER_REGIME = 1000

MIN_SNR_DB = 15.0
MAX_SNR_DB = 30.0

BATCH_SIZE = 32
TRAIN_FRACTION = 0.8

NUM_EPOCHS = 100
LEARNING_RATE = 1e-3

MASK_FLOOR = 0.1

SAVE_PATH = "denoise_autoencoder.pt"


torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0

    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        output = model(noisy)
        loss = criterion(output, clean)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * noisy.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for noisy, clean in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            output = model(noisy)
            loss = criterion(output, clean)

            total_loss += loss.item() * noisy.size(0)

    return total_loss / len(loader.dataset)


print("Generating single-note data...")

single_clean_waveforms, single_noisy_waveforms = generate_waveform_pairs(
    num_examples=NUM_EXAMPLES_PER_REGIME,
    sample_rate=SAMPLE_RATE,
    num_samples=NUM_SAMPLES,
    regime="single_note",
    min_snr_db=MIN_SNR_DB,
    max_snr_db=MAX_SNR_DB,
)

print("Generating multi-note sequence data...")

multi_clean_waveforms, multi_noisy_waveforms = generate_waveform_pairs(
    num_examples=NUM_EXAMPLES_PER_REGIME,
    sample_rate=SAMPLE_RATE,
    num_samples=NUM_SAMPLES,
    regime="multi_note",
    min_snr_db=MIN_SNR_DB,
    max_snr_db=MAX_SNR_DB,
)

print("Generating chord data...")

chord_clean_waveforms, chord_noisy_waveforms = generate_waveform_pairs(
    num_examples=NUM_EXAMPLES_PER_REGIME,
    sample_rate=SAMPLE_RATE,
    num_samples=NUM_SAMPLES,
    regime="chord",
    min_snr_db=MIN_SNR_DB,
    max_snr_db=MAX_SNR_DB,
)

clean_waveforms = torch.cat(
    [
        single_clean_waveforms,
        multi_clean_waveforms,
        chord_clean_waveforms,
    ],
    dim=0,
)

noisy_waveforms = torch.cat(
    [
        single_noisy_waveforms,
        multi_noisy_waveforms,
        chord_noisy_waveforms,
    ],
    dim=0,
)

print("Combined waveform shapes:")
print("clean_waveforms:", clean_waveforms.shape)
print("noisy_waveforms:", noisy_waveforms.shape)

print("Converting waveforms to spectrograms...")

clean_specs, noisy_specs, spec_min, spec_max = waveforms_to_normalized_spectrograms(
    clean_waveforms=clean_waveforms,
    noisy_waveforms=noisy_waveforms,
    n_fft=N_FFT,
    win_length=WIN_LENGTH,
    hop_length=HOP_LENGTH,
)

print("Spectrogram shapes:")
print("clean_specs:", clean_specs.shape)
print("noisy_specs:", noisy_specs.shape)
print("clean range:", clean_specs.min().item(), clean_specs.max().item())
print("noisy range:", noisy_specs.min().item(), noisy_specs.max().item())

train_loader, val_loader, dataset = make_dataloaders(
    noisy_specs=noisy_specs,
    clean_specs=clean_specs,
    batch_size=BATCH_SIZE,
    train_fraction=TRAIN_FRACTION,
)

print(f"Total examples: {len(dataset)}")
print(f"Train examples: {len(train_loader.dataset)}")
print(f"Validation examples: {len(val_loader.dataset)}")

model = DenoiseSpecAutoencoder(mask_floor=MASK_FLOOR).to(device)

def weighted_mse_loss(pred, target, alpha=5.0):
    weight = 1.0 + alpha * target
    return (weight * (pred - target).pow(2)).mean()
criterion = weighted_mse_loss

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

train_losses = []
val_losses = []

print("Starting training...")

for epoch in range(NUM_EPOCHS):
    train_loss = train_one_epoch(
        model=model,
        loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    val_loss = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
    )

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(
        f"Epoch {epoch + 1:03d}/{NUM_EPOCHS} | "
        f"train loss: {train_loss:.6f} | "
        f"val loss: {val_loss:.6f}"
    )


checkpoint = {
    "model_state_dict": model.state_dict(),

    "spec_min": spec_min,
    "spec_max": spec_max,

    "train_losses": train_losses,
    "val_losses": val_losses,

    "config": {
        "sample_rate": SAMPLE_RATE,
        "num_samples": NUM_SAMPLES,

        "n_fft": N_FFT,
        "win_length": WIN_LENGTH,
        "hop_length": HOP_LENGTH,

        "num_examples_per_regime": NUM_EXAMPLES_PER_REGIME,

        "min_snr_db": MIN_SNR_DB,
        "max_snr_db": MAX_SNR_DB,

        "batch_size": BATCH_SIZE,
        "train_fraction": TRAIN_FRACTION,

        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,

        "mask_floor": MASK_FLOOR,
    },
}

torch.save(checkpoint, SAVE_PATH)

print(f"Saved model to: {SAVE_PATH}")