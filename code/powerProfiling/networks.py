from torch import nn
import torch
import snntorch as snn


class CNN_amplitude(nn.Module):
    def __init__(self):
        super().__init__()

        self.B = 64

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32,
                               kernel_size=100, stride=10)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=16,
                               kernel_size=100, stride=10)
        self.fc1 = nn.Linear(in_features=2480, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=9)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.MaxPool1d(kernel_size=2, stride=2)(x)
        x = nn.functional.relu(self.conv2(x))
        x = nn.MaxPool1d(kernel_size=2, stride=2)(x)
        x = x.view(self.B, -1)
        x = nn.Dropout(0.5)(x)
        x = nn.functional.relu(self.fc1(x))
        x = nn.Dropout(0.5)(x)
        x = self.fc2(x)
        return x


class CNN_mfcc(nn.Module):
    def __init__(self):
        super().__init__()

        self.B = 64

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                               kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3)
        self.fc1 = nn.Linear(in_features=5376, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=9)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = nn.functional.relu(self.conv2(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = x.view(self.B, -1)
        x = nn.Dropout(0.5)(x)
        x = nn.functional.relu(self.fc1(x))
        x = nn.Dropout(0.5)(x)
        x = self.fc2(x)
        return x


class SNN_mfcc(nn.Module):
    def __init__(self):
        super().__init__()

        self.time_steps = 345
        self.B = 128

        self.fc1 = nn.Linear(in_features=13, out_features=128)
        self.lif1 = snn.Leaky(beta=0.9, learn_beta=True,
                              threshold=1, learn_threshold=True)

        self.fc2 = nn.Linear(in_features=128, out_features=9)
        self.lif2 = snn.Leaky(beta=0.9, learn_beta=True,
                              threshold=1, learn_threshold=True)

    def forward(self, x):
        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer over time
        mem_rec = []
        spk_rec = []

        # time-loop
        for step in range(self.time_steps):
            cur1 = self.fc1(x[:, :, step])
            assert cur1.shape == (self.B, 128)
            spk1, mem1 = self.lif1(cur1, mem1)
            assert spk1.shape == (self.B, 128)

            cur2 = self.fc2(spk1)
            assert cur2.shape == (self.B, 9)
            spk2, mem2 = self.lif2(cur2, mem2)
            assert spk2.shape == (self.B, 9)

            spk_rec.append(spk2)
            mem_rec.append(mem2)

        spk_rec = torch.stack(spk_rec, dim=0)
        mem_rec = torch.stack(mem_rec, dim=0)

        assert spk_rec.shape == (self.time_steps, self.B, 9)
        assert mem_rec.shape == (self.time_steps, self.B, 9)

        return spk_rec, mem_rec
