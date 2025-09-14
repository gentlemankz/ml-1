#!/usr/bin/env python3
"""
Multi-task Car Inspection Model Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, Tuple
import math

class CarInspectionModel(nn.Module):
    """
    Multi-task model for car inspection:
    - Cleanliness regression (0.0 - 1.0)
    - Damage assessment regression (0.0 - 1.0)
    - Weather condition classification (4 classes)
    """

    def __init__(self,
                 model_name='tf_efficientnetv2_s',
                 pretrained=True,
                 dropout=0.3,
                 num_weather_classes=4):
        super(CarInspectionModel, self).__init__()

        self.model_name = model_name
        self.dropout = dropout

        # Backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool=''  # Remove global pooling
        )

        # Get number of features from backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            if len(features.shape) == 4:  # If spatial dimensions exist
                self.feature_dim = features.shape[1]
                self.has_spatial = True
            else:
                self.feature_dim = features.shape[1]
                self.has_spatial = False

        print(f"🏗️ Backbone: {model_name}")
        print(f"📊 Feature dimension: {self.feature_dim}")
        print(f"🗺️ Has spatial dimensions: {self.has_spatial}")

        # Global pooling if needed
        if self.has_spatial:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.global_pool = nn.Identity()

        # Shared feature processor
        self.feature_processor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2)
        )

        # Task-specific heads

        # Cleanliness head (regression 0-1)
        self.cleanliness_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output 0-1
        )

        # Damage head (regression 0-1)
        self.damage_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output 0-1
        )

        # Weather condition head (classification)
        self.weather_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_weather_classes)
        )

        # Uncertainty heads (for confidence estimation)
        self.cleanliness_uncertainty = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )

        self.damage_uncertainty = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize task-specific head weights"""
        for module in [self.feature_processor,
                      self.cleanliness_head,
                      self.damage_head,
                      self.weather_head,
                      self.cleanliness_uncertainty,
                      self.damage_uncertainty]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, return_uncertainty: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input images [batch_size, 3, H, W]
            return_uncertainty: Whether to return uncertainty estimates

        Returns:
            Dictionary with predictions and optional uncertainties
        """
        batch_size = x.size(0)

        # Extract features
        features = self.backbone(x)

        # Global pooling if needed
        if self.has_spatial:
            features = self.global_pool(features)
            features = features.view(batch_size, -1)

        # Process features
        processed_features = self.feature_processor(features)

        # Task predictions
        cleanliness_score = self.cleanliness_head(processed_features).squeeze(-1)
        damage_score = self.damage_head(processed_features).squeeze(-1)
        weather_logits = self.weather_head(processed_features)

        outputs = {
            'cleanliness_score': cleanliness_score,
            'damage_score': damage_score,
            'weather_logits': weather_logits,
            'weather_probs': F.softmax(weather_logits, dim=1)
        }

        # Add uncertainties if requested
        if return_uncertainty:
            cleanliness_uncertainty = self.cleanliness_uncertainty(processed_features).squeeze(-1)
            damage_uncertainty = self.damage_uncertainty(processed_features).squeeze(-1)

            outputs.update({
                'cleanliness_uncertainty': cleanliness_uncertainty,
                'damage_uncertainty': damage_uncertainty
            })

        return outputs

    def predict_with_confidence(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make predictions with confidence estimates

        Args:
            x: Input images

        Returns:
            Predictions with confidence scores
        """
        outputs = self.forward(x, return_uncertainty=True)

        # Calculate confidence from uncertainty (inverse relationship)
        cleanliness_confidence = torch.sigmoid(-outputs['cleanliness_uncertainty'])
        damage_confidence = torch.sigmoid(-outputs['damage_uncertainty'])

        # Weather confidence from max probability
        weather_confidence = torch.max(outputs['weather_probs'], dim=1)[0]

        # Overall confidence (weighted average)
        overall_confidence = (
            0.4 * cleanliness_confidence +
            0.4 * damage_confidence +
            0.2 * weather_confidence
        )

        outputs.update({
            'cleanliness_confidence': cleanliness_confidence,
            'damage_confidence': damage_confidence,
            'weather_confidence': weather_confidence,
            'overall_confidence': overall_confidence
        })

        return outputs

    def get_interpretable_output(self, x: torch.Tensor) -> Dict:
        """
        Get human-readable interpretation of predictions

        Args:
            x: Input images

        Returns:
            Human-readable predictions
        """
        outputs = self.predict_with_confidence(x)

        # Weather mapping
        weather_names = ['Normal', 'Overcast', 'Snow', 'Rain']
        weather_idx = torch.argmax(outputs['weather_probs'], dim=1)

        # Cleanliness interpretation
        def interpret_cleanliness(score):
            if score >= 0.8:
                return "Very Clean"
            elif score >= 0.6:
                return "Clean"
            elif score >= 0.4:
                return "Slightly Dirty"
            elif score >= 0.2:
                return "Dirty"
            else:
                return "Very Dirty"

        # Damage interpretation
        def interpret_damage(score):
            if score >= 0.8:
                return "No Damage"
            elif score >= 0.6:
                return "Minor Damage"
            elif score >= 0.4:
                return "Moderate Damage"
            elif score >= 0.2:
                return "Significant Damage"
            else:
                return "Severe Damage"

        batch_size = x.size(0)
        interpretations = []

        for i in range(batch_size):
            interpretation = {
                'cleanliness': {
                    'score': float(outputs['cleanliness_score'][i]),
                    'label': interpret_cleanliness(float(outputs['cleanliness_score'][i])),
                    'confidence': float(outputs['cleanliness_confidence'][i])
                },
                'damage': {
                    'score': float(outputs['damage_score'][i]),
                    'label': interpret_damage(float(outputs['damage_score'][i])),
                    'confidence': float(outputs['damage_confidence'][i])
                },
                'weather': {
                    'label': weather_names[int(weather_idx[i])],
                    'confidence': float(outputs['weather_confidence'][i])
                },
                'overall_confidence': float(outputs['overall_confidence'][i])
            }
            interpretations.append(interpretation)

        return interpretations

class MultiTaskLoss(nn.Module):
    """
    Multi-task loss function with learnable task weighting
    """

    def __init__(self,
                 cleanliness_weight=1.0,
                 damage_weight=1.0,
                 weather_weight=0.5,
                 uncertainty_weight=0.1,
                 learnable_weights=True):
        super(MultiTaskLoss, self).__init__()

        self.uncertainty_weight = uncertainty_weight
        self.learnable_weights = learnable_weights

        if learnable_weights:
            # Learnable task weights (log variance technique)
            self.log_vars = nn.Parameter(torch.zeros(3))  # [cleanliness, damage, weather]
        else:
            # Fixed weights
            self.register_buffer('cleanliness_weight', torch.tensor(cleanliness_weight))
            self.register_buffer('damage_weight', torch.tensor(damage_weight))
            self.register_buffer('weather_weight', torch.tensor(weather_weight))

    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss

        Args:
            outputs: Model outputs
            targets: Ground truth targets

        Returns:
            Dictionary with individual and total losses
        """
        losses = {}

        # Regression losses with uncertainty
        if 'cleanliness_uncertainty' in outputs and self.uncertainty_weight > 0:
            # Uncertainty-weighted MSE loss
            precision = torch.exp(-outputs['cleanliness_uncertainty'])
            cleanliness_loss = precision * F.mse_loss(outputs['cleanliness_score'], targets['cleanliness_score'], reduction='none')
            cleanliness_loss += outputs['cleanliness_uncertainty']  # Regularization term
            cleanliness_loss = cleanliness_loss.mean()
        else:
            cleanliness_loss = F.mse_loss(outputs['cleanliness_score'], targets['cleanliness_score'])

        if 'damage_uncertainty' in outputs and self.uncertainty_weight > 0:
            # Uncertainty-weighted MSE loss
            precision = torch.exp(-outputs['damage_uncertainty'])
            damage_loss = precision * F.mse_loss(outputs['damage_score'], targets['damage_score'], reduction='none')
            damage_loss += outputs['damage_uncertainty']  # Regularization term
            damage_loss = damage_loss.mean()
        else:
            damage_loss = F.mse_loss(outputs['damage_score'], targets['damage_score'])

        # Classification loss
        weather_loss = F.cross_entropy(outputs['weather_logits'], targets['weather_condition'])

        losses['cleanliness_loss'] = cleanliness_loss
        losses['damage_loss'] = damage_loss
        losses['weather_loss'] = weather_loss

        # Combine losses
        if self.learnable_weights:
            # Use learnable task weights
            precision_cleanliness = torch.exp(-self.log_vars[0])
            precision_damage = torch.exp(-self.log_vars[1])
            precision_weather = torch.exp(-self.log_vars[2])

            total_loss = (
                precision_cleanliness * cleanliness_loss + self.log_vars[0] +
                precision_damage * damage_loss + self.log_vars[1] +
                precision_weather * weather_loss + self.log_vars[2]
            )

            # Store current weights for monitoring
            losses['weight_cleanliness'] = precision_cleanliness
            losses['weight_damage'] = precision_damage
            losses['weight_weather'] = precision_weather

        else:
            # Use fixed weights
            total_loss = (
                self.cleanliness_weight * cleanliness_loss +
                self.damage_weight * damage_loss +
                self.weather_weight * weather_loss
            )

        losses['total_loss'] = total_loss

        return losses

def create_model(model_name='tf_efficientnetv2_s', pretrained=True, **kwargs):
    """Factory function to create model"""
    return CarInspectionModel(model_name=model_name, pretrained=pretrained, **kwargs)

if __name__ == "__main__":
    # Test model creation and forward pass
    model = create_model('efficientnetv2_s', pretrained=False)

    # Test input
    x = torch.randn(2, 3, 384, 384)

    # Forward pass
    outputs = model.predict_with_confidence(x)

    print("🧪 MODEL TEST:")
    print(f"Input shape: {x.shape}")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")

    # Test interpretable output
    interpretations = model.get_interpretable_output(x)
    print(f"\n📋 SAMPLE INTERPRETATION:")
    print(interpretations[0])

    # Test loss
    targets = {
        'cleanliness_score': torch.rand(2),
        'damage_score': torch.rand(2),
        'weather_condition': torch.randint(0, 4, (2,))
    }

    loss_fn = MultiTaskLoss()
    losses = loss_fn(outputs, targets)

    print(f"\n💰 LOSS TEST:")
    for key, value in losses.items():
        print(f"{key}: {value.item():.4f}")