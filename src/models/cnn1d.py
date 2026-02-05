"""
1D CNN for Tabular IDS Data (CIC-IDS2018-style)
Treat each sample as a 1D “signal” of length d with 1 channel.

TeacherCNN1D (larger capacity):
Input(d) -> [B,1,d]
Conv1D(1->32,k=3) BN ReLU Dropout
Conv1D(32->64,k=3) BN ReLU Dropout
Conv1D(64->128,k=3) BN ReLU
GlobalAvgPool -> embedding [B,128]
Dense(128 -> num_classes)

StudentCNN1D (compressed + prototypes):
Input(d) -> [B,1,d]
Conv1D(1->16,k=3) BN ReLU Dropout
Conv1D(16->32,k=3) BN ReLU
GlobalAvgPool -> features [B,32]
Project(32 -> d_p) BN ReLU -> prototypes [B,d_p]
Dense(d_p -> num_classes)

Correctness output (like your MLP test):
- prints parameter counts, sizes, compression
- prints output shapes for logits/features/prototypes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TeacherCNN1D(nn.Module):
    """Teacher model (larger capacity) for global aggregation."""
    def __init__(self, input_dim: int, num_classes: int, channels=(32, 64, 128)):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.channels = channels

        c1, c2, c3 = channels

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Conv1d(c1, c2, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Conv1d(c2, c3, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm1d(c3),
            nn.ReLU(),
        )

        # Global pooling -> fixed-size embedding regardless of input_dim
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # [B, c3, 1]
        self.classifier = nn.Linear(c3, num_classes)

    def forward(self, x, return_features=False):
        """
        x: [batch_size, input_dim]
        returns:
          logits:   [batch_size, num_classes]
          features: [batch_size, channels[-1]] if return_features=True
        """
        x = x.unsqueeze(1)  # [B, 1, d]
        feats = self.feature_extractor(x)          # [B, c3, d]
        pooled = self.global_pool(feats)           # [B, c3, 1]
        features = pooled.squeeze(-1)              # [B, c3]
        logits = self.classifier(features)         # [B, num_classes]
        if return_features:
            return logits, features
        return logits


class StudentCNN1D(nn.Module):
    """Student model (compressed) for local training + prototype projection."""
    def __init__(self, input_dim: int, num_classes: int, channels=(16, 32), prototype_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.prototype_dim = prototype_dim
        self.channels = channels

        c1, c2 = channels

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Conv1d(c1, c2, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)  # [B, c2, 1]

        # Projection head to prototype space
        self.projection_head = nn.Sequential(
            nn.Linear(c2, prototype_dim),
            nn.BatchNorm1d(prototype_dim),
            nn.ReLU()
        )

        self.classifier = nn.Linear(prototype_dim, num_classes)

    def forward(self, x, return_all=False):
        """
        x: [batch_size, input_dim]
        returns:
          logits:     [batch_size, num_classes]
          prototypes: [batch_size, prototype_dim] if return_all=True
          features:   [batch_size, channels[-1]] if return_all=True (post-pool)
        """
        x = x.unsqueeze(1)                          # [B, 1, d]
        feats = self.feature_extractor(x)           # [B, c2, d]
        pooled = self.global_pool(feats).squeeze(-1)  # [B, c2]
        prototypes = self.projection_head(pooled)   # [B, prototype_dim]
        logits = self.classifier(prototypes)        # [B, num_classes]

        if return_all:
            return logits, prototypes, pooled
        return logits


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing 1D CNN Models...")

    # Parameters (match your MLP test)
    input_dim = 80   # CIC-IDS2018 features
    num_classes = 8
    batch_size = 32

    # Create models
    teacher = TeacherCNN1D(input_dim, num_classes)
    student = StudentCNN1D(input_dim, num_classes, prototype_dim=64)

    # Count parameters
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

    # Test forward pass
    x = torch.randn(batch_size, input_dim)

    # Teacher
    logits_t, features_t = teacher(x, return_features=True)
    print(f"\nTeacher output shapes:")
    print(f" Logits: {logits_t.shape}")
    print(f" Features: {features_t.shape}")

    # Student
    logits_s, protos_s, features_s = student(x, return_all=True)
    print(f"\nStudent output shapes:")
    print(f" Logits: {logits_s.shape}")
    print(f" Prototypes: {protos_s.shape}")
    print(f" Features: {features_s.shape}")

    print("\n Models working correctly!")
