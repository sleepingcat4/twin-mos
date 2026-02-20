import torch
from torch.utils.data import DataLoader

from model import SiameseAudioNet
from loss import MetricRegressionLoss
from data import SingMOSPairs


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():

    dataset = SingMOSPairs(split="train")
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

    model = SiameseAudioNet(embedding_dim=128).to(device)
    criterion = MetricRegressionLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 25
    best_loss = float("inf")

    for epoch in range(epochs):

        model.train()
        total_loss = 0.0

        for audio1, mos1, audio2, mos2 in loader:

            audio1 = audio1.to(device)
            audio2 = audio2.to(device)
            mos1 = mos1.to(device)
            mos2 = mos2.to(device)

            z1, z2 = model(audio1, audio2)
            loss = criterion(z1, z2, mos1, mos2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs}  Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "singmos_metric_best.pt")

    torch.save(model.state_dict(), "singmos_metric_last.pt")
    print("Training complete. Models saved.")


if __name__ == "__main__":
    train()