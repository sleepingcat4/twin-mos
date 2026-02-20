import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class SingMOSPairs(Dataset):
    def __init__(self, split="train", segment_length=16000):
        self.dataset = load_dataset("TangRain/SingMOS", split=split)
        self.n = len(self.dataset)
        self.segment_length = segment_length

    def __len__(self):
        return self.n

    def _process_audio(self, audio_array):
        audio = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)

        if audio.shape[-1] >= self.segment_length:
            start = torch.randint(
                0, audio.shape[-1] - self.segment_length + 1, (1,)
            ).item()
            audio = audio[:, start:start + self.segment_length]
        else:
            pad = self.segment_length - audio.shape[-1]
            audio = torch.nn.functional.pad(audio, (0, pad))

        return audio

    def __getitem__(self, idx):
        idx2 = torch.randint(0, self.n, (1,)).item()

        sample1 = self.dataset[idx]
        sample2 = self.dataset[idx2]

        audio1 = self._process_audio(sample1["audio"]["array"])
        audio2 = self._process_audio(sample2["audio"]["array"])

        mos1 = torch.tensor(sample1["mos"] / 5.0, dtype=torch.float32)
        mos2 = torch.tensor(sample2["mos"] / 5.0, dtype=torch.float32)

        return audio1, mos1, audio2, mos2