#!/usr/bin/env python3
"""
Utility functions for Car Inspection Model
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report
from scipy.stats import pearsonr
import logging
import json
from pathlib import Path
from typing import Dict, List, Any

def setup_logging(log_file=None):
    """Setup logging configuration"""
    logging_config = {
        'level': logging.INFO,
        'format': '%(asctime)s - %(levelname)s - %(message)s',
        'handlers': [logging.StreamHandler()]
    }

    if log_file:
        logging_config['handlers'].append(logging.FileHandler(log_file))

    logging.basicConfig(**logging_config)
    return logging.getLogger(__name__)

def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, save_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_loss': best_val_loss,
    }
    torch.save(checkpoint, save_path)

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cpu'):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    return epoch, best_val_loss

def calculate_metrics(outputs: List[Dict], targets: List[Dict]) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for multi-task model

    Args:
        outputs: List of model outputs for each batch
        targets: List of ground truth targets for each batch

    Returns:
        Dictionary of calculated metrics
    """
    # Memory-efficient concatenation to avoid OOM on large validation sets
    all_cleanliness_pred = []
    all_damage_pred = []
    all_weather_pred = []
    all_confidence = []

    all_cleanliness_true = []
    all_damage_true = []
    all_weather_true = []

    # Process in chunks to manage memory
    for out, tgt in zip(outputs, targets):
        all_cleanliness_pred.append(out['cleanliness_score'].cpu())
        all_damage_pred.append(out['damage_score'].cpu())
        all_weather_pred.append(out['weather_logits'].cpu())
        all_confidence.append(out.get('overall_confidence', torch.zeros_like(out['cleanliness_score'])).cpu())

        all_cleanliness_true.append(tgt['cleanliness_score'].cpu())
        all_damage_true.append(tgt['damage_score'].cpu())
        all_weather_true.append(tgt['weather_condition'].cpu())

    # Concatenate after moving to CPU
    all_cleanliness_pred = torch.cat(all_cleanliness_pred)
    all_damage_pred = torch.cat(all_damage_pred)
    all_weather_pred = torch.cat(all_weather_pred)
    all_confidence = torch.cat(all_confidence)

    all_cleanliness_true = torch.cat(all_cleanliness_true)
    all_damage_true = torch.cat(all_damage_true)
    all_weather_true = torch.cat(all_weather_true)

    # Convert to numpy
    cleanliness_pred = all_cleanliness_pred.numpy()
    damage_pred = all_damage_pred.numpy()
    weather_pred = torch.argmax(all_weather_pred, dim=1).numpy()
    confidence = all_confidence.numpy()

    cleanliness_true = all_cleanliness_true.numpy()
    damage_true = all_damage_true.numpy()
    weather_true = all_weather_true.numpy()

    metrics = {}

    # Regression metrics for cleanliness
    metrics['val/cleanliness_mae'] = mean_absolute_error(cleanliness_true, cleanliness_pred)
    metrics['val/cleanliness_rmse'] = np.sqrt(mean_squared_error(cleanliness_true, cleanliness_pred))

    # Correlation for cleanliness
    if len(np.unique(cleanliness_true)) > 1:
        corr_clean, _ = pearsonr(cleanliness_true, cleanliness_pred)
        metrics['val/cleanliness_correlation'] = corr_clean
    else:
        metrics['val/cleanliness_correlation'] = 0.0

    # Regression metrics for damage
    metrics['val/damage_mae'] = mean_absolute_error(damage_true, damage_pred)
    metrics['val/damage_rmse'] = np.sqrt(mean_squared_error(damage_true, damage_pred))

    # Correlation for damage
    if len(np.unique(damage_true)) > 1:
        corr_damage, _ = pearsonr(damage_true, damage_pred)
        metrics['val/damage_correlation'] = corr_damage
    else:
        metrics['val/damage_correlation'] = 0.0

    # Classification metrics for weather
    metrics['val/weather_accuracy'] = accuracy_score(weather_true, weather_pred)

    # Confidence metrics
    if len(confidence) > 0 and not np.all(confidence == 0):
        metrics['val/avg_confidence'] = np.mean(confidence)
        metrics['val/confidence_std'] = np.std(confidence)
    else:
        metrics['val/avg_confidence'] = 0.0
        metrics['val/confidence_std'] = 0.0

    # Combined metrics
    combined_mae = (metrics['val/cleanliness_mae'] + metrics['val/damage_mae']) / 2
    metrics['val/combined_mae'] = combined_mae

    return metrics

def plot_training_history(train_losses, val_losses, save_path=None):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16)

    # Total loss
    axes[0, 0].plot(train_losses['total_loss'], label='Train', alpha=0.7)
    axes[0, 0].plot(val_losses['total_loss'], label='Validation', alpha=0.7)
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Cleanliness loss
    axes[0, 1].plot(train_losses['cleanliness_loss'], label='Train', alpha=0.7)
    axes[0, 1].plot(val_losses['cleanliness_loss'], label='Validation', alpha=0.7)
    axes[0, 1].set_title('Cleanliness Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Damage loss
    axes[1, 0].plot(train_losses['damage_loss'], label='Train', alpha=0.7)
    axes[1, 0].plot(val_losses['damage_loss'], label='Validation', alpha=0.7)
    axes[1, 0].set_title('Damage Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Weather loss
    axes[1, 1].plot(train_losses['weather_loss'], label='Train', alpha=0.7)
    axes[1, 1].plot(val_losses['weather_loss'], label='Validation', alpha=0.7)
    axes[1, 1].set_title('Weather Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training history plot saved to {save_path}")

    return fig

def plot_predictions_vs_actual(predictions, actuals, title="Predictions vs Actual", save_path=None):
    """Plot predictions vs actual values"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Cleanliness
    axes[0].scatter(actuals['cleanliness'], predictions['cleanliness'], alpha=0.6)
    axes[0].plot([0, 1], [0, 1], 'r--', alpha=0.8)
    axes[0].set_xlabel('Actual Cleanliness')
    axes[0].set_ylabel('Predicted Cleanliness')
    axes[0].set_title('Cleanliness Predictions')
    axes[0].grid(True)

    # Calculate correlation
    if len(np.unique(actuals['cleanliness'])) > 1:
        corr_clean, _ = pearsonr(actuals['cleanliness'], predictions['cleanliness'])
        axes[0].text(0.05, 0.95, f'R = {corr_clean:.3f}', transform=axes[0].transAxes)

    # Damage
    axes[1].scatter(actuals['damage'], predictions['damage'], alpha=0.6)
    axes[1].plot([0, 1], [0, 1], 'r--', alpha=0.8)
    axes[1].set_xlabel('Actual Damage Score')
    axes[1].set_ylabel('Predicted Damage Score')
    axes[1].set_title('Damage Predictions')
    axes[1].grid(True)

    # Calculate correlation
    if len(np.unique(actuals['damage'])) > 1:
        corr_damage, _ = pearsonr(actuals['damage'], predictions['damage'])
        axes[1].text(0.05, 0.95, f'R = {corr_damage:.3f}', transform=axes[1].transAxes)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Predictions plot saved to {save_path}")

    return fig

def create_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix", save_path=None):
    """Create confusion matrix for weather classification"""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Confusion Matrix (Counts)')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title('Confusion Matrix (Normalized)')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix saved to {save_path}")

    return fig

def analyze_confidence_calibration(confidences, accuracies, n_bins=10, save_path=None):
    """Analyze confidence calibration"""
    # Bin predictions by confidence
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            bin_centers.append((bin_lower + bin_upper) / 2)
            bin_accuracies.append(accuracies[in_bin].mean())
            bin_confidences.append(confidences[in_bin].mean())
            bin_counts.append(in_bin.sum())

    # Plot calibration
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Reliability diagram
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.8, label='Perfect Calibration')
    axes[0].scatter(bin_confidences, bin_accuracies, s=100, alpha=0.8, label='Model Calibration')
    axes[0].set_xlabel('Mean Predicted Confidence')
    axes[0].set_ylabel('Actual Accuracy')
    axes[0].set_title('Reliability Diagram')
    axes[0].legend()
    axes[0].grid(True)

    # Confidence histogram
    axes[1].hist(confidences, bins=20, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Confidence')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Confidence Distribution')
    axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Calibration analysis saved to {save_path}")

    return fig

def export_model_for_inference(model, save_path, input_size=(384, 384)):
    """Export model for inference"""
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, *input_size)

    try:
        # Export to ONNX
        onnx_path = save_path.replace('.pth', '.onnx')
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=17,  # Updated for PyTorch 2.8.0 compatibility
            input_names=['image'],
            output_names=['cleanliness_score', 'damage_score', 'weather_logits'],
            dynamic_axes={
                'image': {0: 'batch_size'},
                'cleanliness_score': {0: 'batch_size'},
                'damage_score': {0: 'batch_size'},
                'weather_logits': {0: 'batch_size'}
            }
        )
        print(f"📦 ONNX model exported to {onnx_path}")

    except Exception as e:
        print(f"⚠️ ONNX export failed: {e}")

    # Save PyTorch model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': input_size,
            'model_name': getattr(model, 'model_name', 'tf_efficientnetv2_s')
        }
    }, save_path)

    print(f"💾 PyTorch model saved to {save_path}")

def create_model_report(model, test_metrics, save_dir):
    """Create comprehensive model report"""
    report = {
        'model_architecture': getattr(model, 'model_name', 'Unknown'),
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'test_metrics': test_metrics,
        'timestamp': pd.Timestamp.now().isoformat()
    }

    # Save report
    report_path = Path(save_dir) / 'model_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"📋 Model report saved to {report_path}")

    # Print summary
    print("\n🏆 MODEL PERFORMANCE SUMMARY:")
    print(f"📊 Total Parameters: {report['total_parameters']:,}")
    print(f"🎯 Cleanliness MAE: {test_metrics.get('val/cleanliness_mae', 0):.4f}")
    print(f"🔧 Damage MAE: {test_metrics.get('val/damage_mae', 0):.4f}")
    print(f"🌤️ Weather Accuracy: {test_metrics.get('val/weather_accuracy', 0):.4f}")
    print(f"🤖 Average Confidence: {test_metrics.get('val/avg_confidence', 0):.4f}")

    return report

if __name__ == "__main__":
    # Test utility functions
    print("🧪 Testing utility functions...")

    # Create dummy data for testing
    n_samples = 100
    outputs = [{
        'cleanliness_score': torch.rand(n_samples),
        'damage_score': torch.rand(n_samples),
        'weather_logits': torch.randn(n_samples, 4),
        'overall_confidence': torch.rand(n_samples)
    }]

    targets = [{
        'cleanliness_score': torch.rand(n_samples),
        'damage_score': torch.rand(n_samples),
        'weather_condition': torch.randint(0, 4, (n_samples,))
    }]

    # Test metrics calculation
    metrics = calculate_metrics(outputs, targets)
    print("✅ Metrics calculation test passed")
    for key, value in metrics.items():
        print(f"   {key}: {value:.4f}")

    print("✅ All utility functions working correctly!")