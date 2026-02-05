"""
Lightweight Tabular Transformer for IDS Data

Goal: a small Transformer encoder that treats each feature as a "token".
Works well for tabular data by learning cross-feature interactions via self-attention.

Pattern required (matches your MLP/LSTM/CNN files):
- Feature extractor module
- Projection head (output: prototype_dim)
- Classifier head
- forward() with return_all option
- Test code at the bottom with correctness-style output

Input convention:
- We assume input x is [B, input_dim] (e.g., 80 features).
- We embed each scalar feature into a token vector of dimension d_model:
    tokens: [B, input_dim, d_model]
- Run a small TransformerEncoder over tokens.
- Pool across tokens -> feature vector [B, d_model]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureTokenizer(nn.Module):
    """
    Convert tabular features [B, d] into token embeddings [B, d, d_model].
    Each feature i has its own learnable weight and bias to map scalar -> vector.
    """
    def __init__(self, num_features: int, d_model: int):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model

        # Per-feature affine map: x_i * W_i + b_i
        # W: [d, d_model], b: [d, d_model]
        self.weight = nn.Parameter(torch.randn(num_features, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_features, d_model))

        # Optional feature-wise LayerNorm (stabilizes training)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: [B, d]
        returns tokens: [B, d, d_model]
        """
        # Broadcast multiply: [B, d, 1] * [d, d_model] -> [B, d, d_model]
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        return self.ln(tokens)


class TabTransformerEncoder(nn.Module):
    """
    Lightweight transformer feature extractor:
    tokens [B, d, d_model] -> encoded tokens [B, d, d_model]
    """
    def __init__(self, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,  # small MLP inside the transformer
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Final norm helps stability
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, tokens):
        """
        tokens: [B, d, d_model]
        returns: [B, d, d_model]
        """
        z = self.encoder(tokens)
        return self.out_norm(z)


class TeacherTabTransformer(nn.Module):
    """Teacher transformer (slightly larger)."""
    def __init__(self, input_dim: int, num_classes: int,
                 d_model=96, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.tokenizer = FeatureTokenizer(input_dim, d_model)
        self.feature_extractor = TabTransformerEncoder(
            d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout
        )

        # Pool across tokens -> global feature embedding
        self.pool = nn.AdaptiveAvgPool1d(1)  # will pool along token dimension after transpose
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, return_features=False):
        """
        x: [B, input_dim]
        returns:
          logits: [B, num_classes]
          features: [B, d_model] if return_features=True
        """
        tokens = self.tokenizer(x)                 # [B, d, d_model]
        enc = self.feature_extractor(tokens)       # [B, d, d_model]

        # Pool over tokens dimension d:
        # transpose to [B, d_model, d] so AdaptiveAvgPool1d pools over d
        pooled = self.pool(enc.transpose(1, 2)).squeeze(-1)  # [B, d_model]
        features = pooled
        logits = self.classifier(features)

        if return_features:
            return logits, features
        return logits


class StudentTabTransformer(nn.Module):
    """Student transformer (compressed) + prototype projection."""
    def __init__(self, input_dim: int, num_classes: int,
                 d_model=64, nhead=4, num_layers=2, dropout=0.1,
                 prototype_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.prototype_dim = prototype_dim

        self.tokenizer = FeatureTokenizer(input_dim, d_model)
        self.feature_extractor = TabTransformerEncoder(
            d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.projection_head = nn.Sequential(
            nn.Linear(d_model, prototype_dim),
            nn.BatchNorm1d(prototype_dim),
            nn.ReLU()
        )

        self.classifier = nn.Linear(prototype_dim, num_classes)

    def forward(self, x, return_all=False):
        """
        x: [B, input_dim]
        returns:
          logits:     [B, num_classes]
          prototypes: [B, prototype_dim] if return_all=True
          features:   [B, d_model] if return_all=True
        """
        tokens = self.tokenizer(x)                 # [B, d, d_model]
        enc = self.feature_extractor(tokens)       # [B, d, d_model]
        features = self.pool(enc.transpose(1, 2)).squeeze(-1)  # [B, d_model]

        prototypes = self.projection_head(features)  # [B, prototype_dim]
        logits = self.classifier(prototypes)         # [B, num_classes]

        if return_all:
            return logits, prototypes, features
        return logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -------------------------
# Test the models
# -------------------------
if __name__ == "__main__":
    print("Testing Tabular Transformer Models...")

    # Match your MLP-style test
    input_dim = 80
    num_classes = 8
    batch_size = 32

    teacher = TeacherTabTransformer(input_dim, num_classes)
    student = StudentTabTransformer(input_dim, num_classes, prototype_dim=64)

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

    x = torch.randn(batch_size, input_dim)

    logits_t, features_t = teacher(x, return_features=True)
    print(f"\nTeacher output shapes:")
    print(f" Logits: {logits_t.shape}")
    print(f" Features: {features_t.shape}")

    logits_s, protos_s, features_s = student(x, return_all=True)
    print(f"\nStudent output shapes:")
    print(f" Logits: {logits_s.shape}")
    print(f" Prototypes: {protos_s.shape}")
    print(f" Features: {features_s.shape}")

    print("\n Models working correctly!")
