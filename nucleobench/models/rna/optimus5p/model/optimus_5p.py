from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class Optimus5Prime(nn.Module):
    def __init__(self, weights_path: str | Path | None = None):
        super().__init__()
        # Architecture defined in Sample et al. 2019
        # Input is (Batch, 4, 50)

        # Conv1: 120 filters, Kernel=8, Stride=1, ReLU, Padding='same'
        # Output length: 50
        self.conv1 = nn.Conv1d(
            in_channels=4, out_channels=120, kernel_size=8, stride=1, padding="same"
        )
        self.relu1 = nn.ReLU()

        # Conv2: 120 filters, Kernel=8, Stride=1, ReLU, Padding='same'
        # Output length: 50
        self.conv2 = nn.Conv1d(
            in_channels=120, out_channels=120, kernel_size=8, stride=1, padding="same"
        )
        self.relu2 = nn.ReLU()

        # Conv3: 120 filters, Kernel=8, Stride=1, ReLU, Padding='same'
        # Output length: 50
        self.conv3 = nn.Conv1d(
            in_channels=120, out_channels=120, kernel_size=8, stride=1, padding="same"
        )
        self.relu3 = nn.ReLU()

        self.flatten = nn.Flatten()

        # Calculate flatten size: 120 channels * 50 length = 6000
        self.fc1 = nn.Linear(6000, 40)
        self.relu4 = nn.ReLU()

        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(40, 1)

        if weights_path is None:
            weights_path = Path(__file__).parent / "optimus_5p_weights.pt"
        self.load_or_convert_weights(weights_path)

    def load_or_convert_weights(self, path: str | Path):  # pragma: no cover
        path = Path(path)
        if not path.exists():
            print(
                f"Weights file {path} not found. Attempting to convert from original HDF5..."
            )
            # Local import to avoid circular dependencies and keep dependencies optional if not converting
            import sys

            sys.path.append(str(Path(__file__).parent))
            from convert_weights import convert_weights

            convert_weights(output_path=str(path))

        self.load_weights_from_file(path)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))

        # Permute to (Batch, Length, Channels) to match Keras Flatten order
        x = x.permute(0, 2, 1)

        x = self.flatten(x)
        x = self.dropout(self.relu4(self.fc1(x)))
        x = self.fc2(x)
        return x

    def load_weights_from_file(self, path):  # pragma: no cover
        self.load_state_dict(torch.load(path, weights_only=True))
        self.eval()

    def predict(self, sequences: list[str]) -> np.ndarray:
        """
        Takes a list of sequence strings, one-hot encodes them,
        runs inference, and returns MRL scores as a numpy array.
        """
        device = next(self.parameters()).device
        encoded = self._encode_sequences(sequences)
        tensor = torch.tensor(encoded, dtype=torch.float32).to(device)

        with torch.no_grad():
            output = self.forward(tensor)

        return output.cpu().numpy().flatten()

    def _encode_sequences(self, sequences: list[str], length: int = 50) -> np.ndarray:
        """
        One-hot encodes sequences to (N, 4, Length).
        A -> [1,0,0,0]
        C -> [0,1,0,0]
        G -> [0,0,1,0]
        T/U -> [0,0,0,1]
        Pads with N ([0,0,0,0]) if too short, truncates if too long.
        """
        # Map: A, C, G, T
        # PyTorch Conv1d expects (Batch, Channels, Length)

        n = len(sequences)
        one_hot = np.zeros((n, 4, length), dtype=np.float32)

        map_dict = {
            "A": 0,
            "a": 0,
            "C": 1,
            "c": 1,
            "G": 2,
            "g": 2,
            "T": 3,
            "t": 3,
            "U": 3,
            "u": 3,
        }

        for i, seq in enumerate(sequences):
            # Truncate if too long
            seq = seq[:length]
            for j, char in enumerate(seq):
                if char in map_dict:
                    one_hot[i, map_dict[char], j] = 1.0

        return one_hot
