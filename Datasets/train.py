import torch
import torch.nn as nn
import torch.optim as optim
import os
from GAN import Generator, Discriminator
from torchvision.transforms import Resize

# Set parameters
latent_dim = 100
img_size = (126, 256)  # Spectrogram dimensions
batch_size = 11  # Adjust as per your GPU memory
epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Generator and Discriminator
generator = Generator(latent_dim=latent_dim, img_size=img_size).to(device)
discriminator = Discriminator(img_size=img_size).to(device)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function
criterion = nn.BCELoss()

# Directories for saving outputs
save_dir = "Generated_Spectrograms"
os.makedirs(save_dir, exist_ok=True)

# Training Loop
for epoch in range(epochs):
    for batch_idx in range(5):  # Simulating batches for simplicity; replace with a real DataLoader
        # Generate real spectrograms (dummy data for illustration)
        real_spectrograms = torch.randn(batch_size, 1, *img_size).to(device)  # Replace with real data
        real_labels = torch.ones(batch_size, 1).to(device)  # Real labels

        # Train Generator
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_spectrograms = generator(z)
        fake_labels = torch.ones(batch_size, 1).to(device)  # Fool discriminator
        g_loss = criterion(discriminator(fake_spectrograms), fake_labels)
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = criterion(discriminator(real_spectrograms), real_labels)
        fake_labels = torch.zeros(batch_size, 1).to(device)  # Correctly label fake samples
        fake_loss = criterion(discriminator(fake_spectrograms.detach()), fake_labels)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # Logging
        if batch_idx % 2 == 0:
            print(f"Epoch [{epoch}/{epochs}] Batch {batch_idx} "
                  f"Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}")

    # Save generated samples at the end of each epoch
    z = torch.randn(1, latent_dim).to(device)  # Generate one sample
    generated_spectrogram = generator(z).detach().cpu()
    save_path = os.path.join(save_dir, f"generated_epoch_{epoch}.pt")
    torch.save(generated_spectrogram, save_path)


print("Training complete!")
