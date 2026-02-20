import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from tqdm import tqdm

class SingMOSPairs(Dataset):
    def __init__(self, split="train", segment_length=16000):
        print("Loading SingMOS dataset...")
        # Will use cached data if available; downloads only if missing
        raw = load_dataset(
            "TangRain/SingMOS",
            split=split,
            download_mode="reuse_dataset_if_exists"
        )
        raw = raw.with_format("numpy")

        self.segment_length = segment_length
        self.audio = []
        self.mos = []

        print(f"Preloading {len(raw)} audio samples into memory...")
        for sample in tqdm(raw, desc="Loading audio"):
            self.audio.append(torch.tensor(sample["audio"]["array"], dtype=torch.float32))
            self.mos.append(torch.tensor(sample["mos"] / 5.0, dtype=torch.float32))  # normalize 0–1

        self.n = len(self.audio)
        print("Preloading complete.")

    def __len__(self):
        return self.n

    def _process_audio(self, audio):
        audio = audio.unsqueeze(0)  # add channel dimension [1, T]

        if audio.shape[-1] >= self.segment_length:
            start = torch.randint(0, audio.shape[-1] - self.segment_length + 1, (1,)).item()
            audio = audio[:, start:start + self.segment_length]
        else:
            pad = self.segment_length - audio.shape[-1]
            audio = torch.nn.functional.pad(audio, (0, pad))

        return audio

    def __getitem__(self, idx):
        # Random pairing
        idx2 = torch.randint(0, self.n, (1,)).item()

        audio1 = self._process_audio(self.audio[idx])
        audio2 = self._process_audio(self.audio[idx2])

        mos1 = self.mos[idx]
        mos2 = self.mos[idx2]

        return audio1, mos1, audio2, mos2
