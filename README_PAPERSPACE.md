# 🚀 Car Inspection ML - Paperspace Training Guide

## 📋 Quick Start (3 steps to training!)

### 1. Setup Paperspace Instance
```bash
# Create Paperspace GPU instance:
# - Machine: RTX 4000, RTX 5000, or A100
# - Template: PyTorch 2.0
# - Auto-shutdown: 2 hours

# Upload your project folder to Paperspace
```

### 2. Install & Run
```bash
# Clone or upload the ML folder to Paperspace
cd /notebooks/ml  # or your project path

# Run the complete pipeline (one command!)
./run_paperspace.sh
```

### 3. Monitor & Results
- Check Weights & Biases dashboard for real-time metrics
- Models saved in `models/` directory
- Inference-ready model exported automatically

---

## 🔧 Manual Step-by-Step (if needed)

### Environment Setup
```bash
python3 setup_paperspace.py
pip install -r requirements.txt
```

### Data Preparation
```bash
python3 convert_annotations.py
python3 dataset.py  # Test loading
```

### Training
```bash
python3 train.py \
    --model_name efficientnetv2_l \
    --batch_size 32 \
    --epochs 30 \
    --lr 1e-4 \
    --mixed_precision \
    --wandb_project car-inspection-hackathon
```

---

## 📊 Expected Results

### Training Metrics (Target)
- **Cleanliness MAE:** < 0.15
- **Damage MAE:** < 0.12
- **Weather Accuracy:** > 85%
- **Training Time:** 3-5 hours (RTX 4000)

### Dataset Stats
- **Total Images:** 5,297
- **COCO Annotated:** 5,282
- **Weather Conditions:** 15 special cases
- **Multi-task Labels:** Cleanliness + Damage + Weather

---

## 🎯 Model Architecture

```python
# Single Multi-Task Model
EfficientNetV2-L Backbone
├── Cleanliness Head (Regression 0-1)
├── Damage Head (Regression 0-1)
├── Weather Head (Classification 4 classes)
└── Uncertainty Estimation

# Output Example
{
    "cleanliness_score": 0.85,    # 0=очень грязно, 1=чисто
    "damage_score": 0.92,         # 0=сильно битый, 1=целый
    "weather_condition": "rain",   # normal/overcast/snow/rain
    "confidence": 0.88            # общая уверенность
}
```

---

## 💰 Paperspace Cost Estimation

### Training Options
- **RTX 4000:** $0.51/hour → ~$2.50 total
- **RTX 5000:** $0.78/hour → ~$3.90 total
- **A100:** $3.18/hour → ~$15.90 total (fastest)

### Recommended for Hackathon
**RTX 4000** - Best cost/performance balance

---

## 🐛 Troubleshooting

### Common Issues
1. **Out of Memory:** Reduce batch_size to 16 or 8
2. **Data Loading Slow:** Reduce num_workers
3. **Training Crashes:** Use CPU if GPU fails

### Debug Commands
```bash
# Check GPU
nvidia-smi

# Test dataset
python3 -c "from dataset import analyze_dataset; analyze_dataset('data/car_inspection_dataset.csv')"

# Test model
python3 -c "from model import create_model; model = create_model('efficientnetv2_s'); print('Model OK')"
```

---

## 📈 Monitoring Training

### Weights & Biases Dashboard
- Real-time loss curves
- Learning rate scheduling
- Model predictions visualization
- Hardware utilization

### Key Metrics to Watch
- `train/total_loss` - Should decrease steadily
- `val/cleanliness_mae` - Target < 0.15
- `val/damage_mae` - Target < 0.12
- `val/weather_accuracy` - Target > 0.85

---

## 🎉 After Training

### Export for Inference
```bash
# Automatically done by run_paperspace.sh
# Manual export:
python3 export_model.py models/experiment_name/best_model.pth
```

### Model Files Generated
```
models/experiment_name/
├── best_model.pth           # Best validation model
├── best_model_inference.pth # Ready for inference
├── best_model.onnx         # ONNX format (if supported)
├── checkpoint.pth          # Latest checkpoint
└── model_report.json       # Performance summary
```

---

## 🏆 Competition Advantages

### Technical Innovations
- 🎯 **Regression-based scoring** (not just binary)
- 🌦️ **Weather-aware predictions**
- 🔥 **Multi-level damage assessment**
- ⚡ **Uncertainty quantification**
- 🧠 **Single unified model**

### Business Value
- 📊 **Nuanced scoring** (0-100% instead of yes/no)
- 🤖 **Confidence estimates** for human review
- 🌍 **Weather adaptation** for different conditions
- ⚡ **Fast inference** for real-time use

---

## 🎤 For Presentation

### Key Results to Show
1. **Baseline vs Our Model** comparison
2. **Multi-level predictions** examples
3. **Weather condition** handling
4. **Confidence calibration** graphs
5. **Real-time inference** demo

### Demo Script
```python
# Load trained model
from model import CarInspectionModel
model = CarInspectionModel()
checkpoint = torch.load('models/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Predict on new image
predictions = model.get_interpretable_output(image_tensor)
print(predictions[0])  # Human-readable results
```

---

**🚀 Ready to win the hackathon! Good luck!**