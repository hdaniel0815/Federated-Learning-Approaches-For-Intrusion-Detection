"""
LSTM for Temporal IDS Data

Architecture (Teacher):
Input [B, T, d]
LSTM(d -> hidden) (stacked)
Last hidden state = temporal feature
Dense(hidden -> num_classes)

Architecture (Student):
Input [B, T, d]
LSTM(d -> hidden_small)
Last hidden state = temporal feature
Project(hidden_small -> prototype_dim)
Dense(prototype_dim -> num_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TeacherLSTM(nn.Module):
    """Teacher LSTM model for temporal modeling."""
    def __init__(self, input_dim: int, num_classes: int,
                 hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        self.feature_extractor = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, return_features=False):
        """
        x: [batch, seq_len, input_dim]
        """
        _, (h_n, _) = self.feature_extractor(x)
        features = h_n[-1]              # [B, hidden_dim]
        logits = self.classifier(features)

        if return_features:
            return logits, features
        return logits


class StudentLSTM(nn.Module):
    """Student LSTM model with prototype projection."""
    def __init__(self, input_dim: int, num_classes: int,
                 hidden_dim=64, num_layers=1, prototype_dim=64):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.prototype_dim = prototype_dim

        self.feature_extractor = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, prototype_dim),
            nn.BatchNorm1d(prototype_dim),
            nn.ReLU()
        )

        self.classifier = nn.Linear(prototype_dim, num_classes)

    def forward(self, x, return_all=False):
        """
        x: [batch, seq_len, input_dim]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)          # (D,) -> (1, D)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        _, (h_n, _) = self.feature_extractor(x)
        features = h_n[-1]                    # [B, hidden_dim]
        prototypes = self.projection_head(features)
        logits = self.classifier(prototypes)

        if return_all:
            return logits, prototypes, features
        return logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -------------------------
# Test the models
# -------------------------
if __name__ == "__main__":
    print("Testing LSTM Models...")

    # Example parameters
    batch_size = 32
    seq_len = 10          # number of time steps
    input_dim = 8         # features per timestep
    num_classes = 8

    teacher = TeacherLSTM(input_dim, num_classes)
    student = StudentLSTM(input_dim, num_classes)

    teacher_params = count_parameters(teacher)
    student_params = count_parameters(student)

    print(f"\nTeacher Model:")
    print(f" Parameters: {teacher_params:,}")
    print(f" Size: {teacher_params * 4 / (1024**2):.2f} MB")

    print(f"\nStudent Model:")
    print(f" Parameters: {student_params:,}")
    print(f" Size: {student_params * 4 / (1024**2):.2f} MB")

    compression = (1 - student_params / teacher_params) * 100
    print(f"\nCompression: {compression:.1f}%")

    # Dummy temporal input
    x = torch.randn(batch_size, seq_len, input_dim)

    # Teacher forward
    logits_t, features_t = teacher(x, return_features=True)
    print(f"\nTeacher output shapes:")
    print(f" Logits: {logits_t.shape}")
    print(f" Features: {features_t.shape}")

    # Student forward
    logits_s, protos_s, features_s = student(x, return_all=True)
    print(f"\nStudent output shapes:")
    print(f" Logits: {logits_s.shape}")
    print(f" Prototypes: {protos_s.shape}")
    print(f" Features: {features_s.shape}")

    print("\n Models working correctly!")
