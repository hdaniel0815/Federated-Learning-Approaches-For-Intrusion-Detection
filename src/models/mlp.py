"""
MLP (Multi-Layer Perceptron) for Tabular IDS Data
Architecture:
Input(d) Dense(256) BatchNorm ReLU Dropout(0.3)
Dense(128) BatchNorm ReLU Dropout(0.3)
Dense(64) BatchNorm ReLU [embedding layer]
Project(64 d_p) [prototype projection]
Dense(d_p num_classes) Softmax
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
class TeacherMLP(nn.Module):
    """Teacher model (larger capacity) for global aggregation."""
    def __init__(self, input_dim: int, num_classes: int, hidden_dims=[256, 128, 64]):
        super(TeacherMLP, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims

        # Feature extractor layers
        layers = []
        in_dim = input_dim
        for i, out_dim in enumerate(hidden_dims):
            layers.extend([
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(0.3 if i < len(hidden_dims) - 1 else 0.2)
            ])
            in_dim = out_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Classifier head
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)


    def forward(self, x, return_features=False):
        """
        Forward pass.
        Args:
        x: Input tensor [batch_size, input_dim]
        return_features: If True, return intermediate features
        Returns:
        logits: [batch_size, num_classes]
        features: [batch_size, hidden_dims[-1]] (if return_features=True)
        """
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


class StudentMLP(nn.Module):
    """Student model (compressed) for local training."""
    def __init__(self, input_dim: int, num_classes: int,
    hidden_dims=[128, 64, 32], prototype_dim=64):
        super(StudentMLP, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.prototype_dim = prototype_dim
        # Feature extractor
        layers = []
        in_dim = input_dim
        for i, out_dim in enumerate(hidden_dims):
            layers.extend([
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(0.3 if i < len(hidden_dims) - 1 else 0.2)
            ])
            in_dim = out_dim
        self.feature_extractor = nn.Sequential(*layers)
        # Projection head (to prototype space)
        self.projection_head = nn.Sequential(
        nn.Linear(hidden_dims[-1], prototype_dim),
        nn.BatchNorm1d(prototype_dim),
        nn.ReLU()
        )
        
        # Classifier head
        self.classifier = nn.Linear(prototype_dim, num_classes)


    def forward(self, x, return_all=False):
        """
        Forward pass with optional prototype extraction.
        Args:
        x: Input [batch, input_dim]
        return_all: If True, return (logits, prototypes, features)
        Returns:
        logits: [batch, num_classes]
        prototypes: [batch, prototype_dim] (if return_all=True)
        features: [batch, hidden_dims[-1]] (if return_all=True)
        """
        features = self.feature_extractor(x)
        prototypes = self.projection_head(features)
        logits = self.classifier(prototypes)
        if return_all:
            return logits, prototypes, features
        return logits
    

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Test the models
if __name__ == "__main__":
    print("Testing MLP Models...")
    # Parameters
    input_dim = 80 # CIC-IDS2018 features
    num_classes = 8
    batch_size = 32
    # Create models
    teacher = TeacherMLP(input_dim, num_classes)
    student = StudentMLP(input_dim, num_classes)
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