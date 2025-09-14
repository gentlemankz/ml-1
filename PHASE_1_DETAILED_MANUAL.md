# 🚀 PHASE 1: DETAILED MANUAL - Azure ML Setup for Car Damage Detection

**Goal:** Set up modern Azure ML infrastructure and start baseline training in 2-4 hours (with high-performance compute)
**Focus:** ML ONLY (FastAPI, NextJS, deployment later)
**Performance:** Optimized for 12-hour hackathon with maximum speed compute

---

## 📋 PHASE 1 CHECKLIST (0-4 hours)

- [ ] **Step 1:** Azure ML Workspace Setup (30 min)
- [ ] **Step 2:** Environment & Dependencies (30 min)
- [ ] **Step 3:** Dataset Preparation (60 min)
- [ ] **Step 4:** Baseline Model Training (90 min)
- [ ] **Step 5:** Model Evaluation (30 min)

---

## 🌐 STEP 1: AZURE ML WORKSPACE SETUP (30 minutes)

### **1.1 Access Azure Portal**
1. Go to **https://portal.azure.com**
2. Login with your Microsoft for Startups Founders Hub account
3. Verify you see your subscription (should show startup benefits)

### **1.2 Create Resource Group**
1. Click **"Resource groups"** → **"+ Create"**
2. **Subscription:** Select your Founders Hub subscription
3. **Resource group name:** `hackathon-car-inspection-rg`
4. **Region:** `East US` (has GPU quotas)
5. Click **"Review + Create"** → **"Create"**

### **1.3 Create Azure Machine Learning Workspace**
1. Search **"Machine Learning"** in Azure Portal
2. Click **"+ Create"** → **"New workspace"**
3. Fill details:
   ```
   Subscription: [Your Founders Hub subscription]
   Resource group: hackathon-car-inspection-rg
   Workspace name: car-inspection-ml-workspace
   Region: East US
   Storage account: [Auto-created]
   Key vault: [Auto-created]
   Application insights: [Auto-created]
   Container registry: [Auto-created]
   ```
4. Click **"Review + Create"** → **"Create"**
5. **Wait 3-5 minutes** for deployment

### **1.4 Access Azure ML Studio**
1. After deployment, click **"Go to resource"**
2. Click **"Launch studio"** button
3. You'll be redirected to **https://ml.azure.com**
4. **IMPORTANT:** This is the NEW Azure ML Studio (not the deprecated Classic)

---

## 🛠 STEP 2: ENVIRONMENT & DEPENDENCIES (30 minutes)

### **2.1 Create High-Performance Compute Instance (for development)**
1. In Azure ML Studio, go to **"Compute"** → **"Compute instances"**
2. Click **"+ New"**
3. **For Maximum Speed (Hackathon):**
   ```
   Compute name: hackathon-dev-powerhouse
   Virtual machine type: CPU
   Virtual machine size: Standard_D32s_v5 (32 cores, 128GB RAM)
   Cost: ~$1.50/hour
   ```
4. **Budget Option:**
   ```
   Compute name: car-inspection-dev
   Virtual machine type: CPU
   Virtual machine size: Standard_DS3_v2 (4 cores, 14GB RAM)
   ```
5. **Advanced Settings:**
   - Enable SSH: ✅
   - Idle shutdown: 60 minutes
   - Assign managed identity: ✅
6. Click **"Create"** (takes 2-3 minutes)

### **2.2 Create Maximum Performance GPU Compute Cluster (for training)**
1. Go to **"Compute"** → **"Compute clusters"**
2. Click **"+ New"**
3. **FASTEST Option (A100 GPU):**
   ```
   Compute name: hackathon-gpu-beast
   Virtual machine type: GPU
   Virtual machine size: Standard_NC24ads_A100_v4 (24 cores, 220GB RAM, 1 A100 40GB)
   Virtual machine priority: Dedicated (fastest startup)
   Min nodes: 1 (keep warm, no startup delay)
   Max nodes: 2
   Idle seconds before scale down: 3600 (1 hour)
   Cost: ~$3.20/hour
   ```
4. **Alternative High-Performance Options (if A100 unavailable):**
   ```
   Option 1: Standard_NC12s_v3 (12 cores, 224GB RAM, 2x Tesla V100)
   Option 2: Standard_NC6s_v3 (6 cores, 112GB RAM, 1x Tesla V100)
   Option 3: Standard_NC4as_T4_v3 (4 cores, 28GB RAM, 1x Tesla T4)
   ```
5. **Regions to Try (in order):**
   - East US (try first)
   - West US 2
   - West Europe
   - UK South
6. Click **"Create"** (takes 5-10 minutes)

### **2.3 Set Up Development Environment**
1. Wait for compute instance to be **"Running"**
2. Click **"Terminal"** next to your compute instance
3. Run setup commands:
```bash
# Clone your repository (if you have one)
# git clone https://github.com/your-repo/car-inspection.git
# cd car-inspection

# Create working directory
mkdir -p car-inspection-ml
cd car-inspection-ml

# Install Azure ML SDK v2
pip install azure-ai-ml==1.15.0 azure-identity==1.15.0

# Install ML dependencies with optimizations
pip install torch torchvision timm albumentations opencv-python pandas matplotlib
pip install mlflow azure-ai-ml accelerate

# For maximum performance training
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 📊 STEP 3: DATASET PREPARATION (60 minutes)

### **3.1 Download Provided Datasets**
1. In terminal, download the competition datasets:
```bash
# Create data directory
mkdir -p data/raw

# Download from provided URLs
wget -O data/raw/rust_scratch.zip "https://universe.roboflow.com/seva-at1qy/rust-and-scrach/dataset/download"
wget -O data/raw/car_scratch_dent.zip "https://universe.roboflow.com/carpro/car-scratch-and-dent/dataset/download"

# Unzip datasets
cd data/raw
unzip rust_scratch.zip -d rust_scratch/
unzip car_scratch_dent.zip -d car_scratch_dent/
cd ../..
```

### **3.2 Create Custom Cleanliness Dataset**
1. Create data structure:
```bash
mkdir -p data/processed/cleanliness/{clean,dirty}
mkdir -p data/processed/damage/{damaged,undamaged}
```

2. **MANUAL TASK:** Collect 300-400 car photos for cleanliness dataset:
   - Search online for clean car images (150-200 photos)
   - Search for dirty/muddy car images (150-200 photos)
   - Save to respective folders
   - **Tip:** Use search terms: "clean car", "dirty car", "muddy vehicle"

### **3.3 Data Annotation Script**
Create `prepare_data.py`:
```python
import os
import pandas as pd
from pathlib import Path

def create_dataset_csv():
    """Create CSV files for training"""

    # Cleanliness dataset
    clean_images = list(Path("data/processed/cleanliness/clean").glob("*.jpg"))
    dirty_images = list(Path("data/processed/cleanliness/dirty").glob("*.jpg"))

    cleanliness_data = []

    # Clean images (score 0.8-1.0)
    for img_path in clean_images:
        cleanliness_data.append({
            'image_path': str(img_path),
            'cleanliness_score': 0.9,  # High cleanliness
            'is_clean': 1,
            'category': 'clean'
        })

    # Dirty images (score 0.0-0.4)
    for img_path in dirty_images:
        cleanliness_data.append({
            'image_path': str(img_path),
            'cleanliness_score': 0.2,  # Low cleanliness
            'is_clean': 0,
            'category': 'dirty'
        })

    # Save to CSV
    df = pd.DataFrame(cleanliness_data)
    df.to_csv('data/cleanliness_dataset.csv', index=False)
    print(f"Created cleanliness dataset with {len(df)} samples")

if __name__ == "__main__":
    create_dataset_csv()
```

### **3.4 Upload Data to Azure ML**
1. In Azure ML Studio, go to **"Data"** → **"Datastores"**
2. Click on **"workspaceblobstore"** (default datastore)
3. Click **"Upload"** → **"Upload folder"**
4. Select your `data/` folder and upload

---

## 🤖 STEP 4: BASELINE MODEL TRAINING (90 minutes)

### **4.1 Create Training Script**
Create `train_baseline.py`:
```python
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
import pandas as pd
from PIL import Image
import mlflow
import argparse
from pathlib import Path

class CarInspectionDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['image_path']

        # Load image
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Labels
        cleanliness_score = torch.tensor(row['cleanliness_score'], dtype=torch.float32)
        is_clean = torch.tensor(row['is_clean'], dtype=torch.long)

        return image, cleanliness_score, is_clean

def create_model(num_classes=2):
    """Create EfficientNetV2 model"""
    model = timm.create_model('efficientnetv2_s', pretrained=True)

    # Replace classifier for multi-task learning
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes + 1)  # +1 for regression score
    )

    return model

def train_epoch(model, train_loader, optimizer, criterion_class, criterion_reg, device, use_mixed_precision=True):
    model.train()
    total_loss = 0

    # Mixed precision for 2x speed on modern GPUs
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler() if use_mixed_precision else None

    for images, cleanliness_scores, is_clean in train_loader:
        images, cleanliness_scores, is_clean = images.to(device, non_blocking=True), cleanliness_scores.to(device, non_blocking=True), is_clean.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_mixed_precision and scaler:
            with autocast():
                outputs = model(images)
                # Split outputs
                classification_out = outputs[:, :2]  # First 2 for classification
                regression_out = outputs[:, 2]       # Last 1 for regression
                # Losses
                class_loss = criterion_class(classification_out, is_clean)
                reg_loss = criterion_reg(regression_out, cleanliness_scores)
                total_loss_batch = class_loss + reg_loss

            scaler.scale(total_loss_batch).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            # Split outputs
            classification_out = outputs[:, :2]  # First 2 for classification
            regression_out = outputs[:, 2]       # Last 1 for regression
            # Losses
            class_loss = criterion_class(classification_out, is_clean)
            reg_loss = criterion_reg(regression_out, cleanliness_scores)
            total_loss_batch = class_loss + reg_loss
            total_loss_batch.backward()
            optimizer.step()

        total_loss += total_loss_batch.item()

    return total_loss / len(train_loader)

def main():
    # MLflow tracking
    mlflow.autolog()

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset and DataLoader (optimized for high-performance compute)
    dataset = CarInspectionDataset('data/cleanliness_dataset.csv', transform=train_transform)

    # Batch size optimization based on GPU
    if 'A100' in str(device) or torch.cuda.get_device_properties(0).total_memory > 30e9:
        batch_size = 64  # A100 or high-memory GPU
    elif 'V100' in str(device) or torch.cuda.get_device_properties(0).total_memory > 15e9:
        batch_size = 48  # V100 or medium-memory GPU
    else:
        batch_size = 32  # T4 or smaller GPU

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,  # Use multiple CPU cores
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True  # Keep workers alive
    )

    # Model
    model = create_model().to(device)

    # Loss and optimizer
    criterion_class = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop (optimized for hackathon speed)
    num_epochs = 15 if torch.cuda.is_available() else 10  # Fewer epochs with powerful GPU
    use_mixed_precision = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7  # Tensor cores

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion_class, criterion_reg, device, use_mixed_precision)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}")

        # Log metrics
        mlflow.log_metric("train_loss", train_loss, step=epoch)

    # Save model
    torch.save(model.state_dict(), "baseline_model.pth")
    mlflow.pytorch.log_model(model, "model")

    print("Training completed!")

if __name__ == "__main__":
    main()
```

### **4.2 Create Azure ML Job**
Create `submit_training_job.py`:
```python
from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

def submit_training_job():
    # Initialize ML Client
    credential = DefaultAzureCredential()
    ml_client = MLClient.from_config(credential=credential)  # Reads from config

    # Define environment
    env = Environment(
        name="car-inspection-env",
        image="mcr.microsoft.com/azureml/curated/acpt-pytorch-2.0-cuda11.7:latest",
        conda_file="environment_v2.yml"
    )

    # Define training job
    job = command(
        experiment_name="car-inspection-baseline",
        code="./",  # Current directory
        command="python train_baseline.py",
        environment=env,
        compute="hackathon-gpu-beast",  # Use high-performance cluster
        display_name="Baseline EfficientNetV2-S Training"
    )

    # Submit job
    returned_job = ml_client.jobs.create_or_update(job)

    print(f"Job submitted: {returned_job.studio_url}")
    print(f"Job name: {returned_job.name}")

if __name__ == "__main__":
    submit_training_job()
```

### **4.3 Submit Training Job**
1. In terminal, run:
```bash
# Create ML client config
az login  # Login with your Azure account
az ml workspace show --name car-inspection-ml-workspace --resource-group hackathon-car-inspection-rg

# Submit training job
python submit_training_job.py
```

2. **Monitor Training:**
   - Go to Azure ML Studio → **"Jobs"**
   - Click on your submitted job
   - Monitor real-time logs and metrics

---

## 📈 STEP 5: MODEL EVALUATION (30 minutes)

### **5.1 Check Training Results**
1. In Azure ML Studio, go to **"Jobs"** → Click your job
2. Go to **"Metrics"** tab to see training loss
3. Go to **"Outputs + logs"** → **"logs"** → **"user_logs"** → **"std_log.txt"**

### **5.2 Download Trained Model**
1. In your job page, go to **"Outputs + logs"**
2. Download `baseline_model.pth`

### **5.3 Quick Local Testing**
Create `test_model.py`:
```python
import torch
import timm
from PIL import Image
from torchvision import transforms

def load_model(model_path):
    model = timm.create_model('efficientnetv2_s', pretrained=False)
    num_features = model.classifier.in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.2),
        torch.nn.Linear(num_features, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 3)  # 2 classes + 1 regression
    )

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def test_image(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        classification = torch.softmax(outputs[:, :2], dim=1)
        regression_score = torch.sigmoid(outputs[:, 2])

    clean_prob = classification[0][1].item()
    cleanliness_score = regression_score[0].item()

    print(f"Clean probability: {clean_prob:.3f}")
    print(f"Cleanliness score: {cleanliness_score:.3f}")

if __name__ == "__main__":
    model = load_model("baseline_model.pth")
    # Test with a sample image
    test_image(model, "path/to/test/image.jpg")
```

---

## ✅ PHASE 1 SUCCESS CRITERIA

After 4 hours, you should have:

- [ ] ✅ Working Azure ML workspace
- [ ] ✅ GPU compute cluster configured
- [ ] ✅ Custom cleanliness dataset (300+ images)
- [ ] ✅ Baseline EfficientNetV2-S model trained
- [ ] ✅ Training metrics visible in Azure ML Studio
- [ ] ✅ Model successfully saved and downloadable
- [ ] ✅ Basic inference testing working

### **Expected Results:**
- **Training Loss:** Should decrease to < 1.0
- **Model Size:** ~50MB
- **Training Time with High-Performance Setup:**
  - A100 GPU: 15-20 minutes
  - V100 GPU: 30-45 minutes
  - T4 GPU: 60-90 minutes
- **Clean/Dirty Accuracy:** >80% (baseline)
- **Total Phase 1 Time:** 2-3 hours (with A100) vs 4+ hours (standard)

---

## 💰 HACKATHON COST ESTIMATE (12 hours)

### **High-Performance Setup:**
- A100 GPU cluster: ~$40
- High-end CPU instance: ~$18
- Storage & misc: ~$5
- **Total: ~$65 for maximum speed**

### **Budget Setup:**
- V100 GPU cluster: ~$20
- Standard CPU instance: ~$8
- Storage & misc: ~$3
- **Total: ~$35 for good performance**

## 🚀 NEXT STEPS (After Phase 1)

1. **Phase 2:** Advanced model training (EfficientNetV2-L, ensemble)
2. **Phase 3:** Model optimization (quantization, ONNX export)
3. **Phase 4:** FastAPI integration
4. **Phase 5:** NextJS frontend integration
5. **Phase 6:** Production deployment

---

## 🛟 TROUBLESHOOTING

### **Common Issues:**

1. **GPU Quota Error:**
   - Try regions in this order: West US 2, West Europe, UK South, North Central US
   - Start with smaller GPU: Standard_NC4as_T4_v3 → Standard_NC6s_v3 → Standard_NC12s_v3
   - Contact Azure support via chat (mention hackathon urgency)
   - Consider multiple smaller instances instead of one large one

2. **Dataset Upload Fails:**
   - Use Azure Storage Explorer
   - Upload in smaller batches

3. **Training Job Fails:**
   - Check logs in "Outputs + logs"
   - Verify dataset paths
   - Reduce batch size to 16

4. **Authentication Issues:**
   - Run `az login` again
   - Check subscription access in Azure Portal

### **Getting Help:**
- Azure ML Documentation: https://docs.microsoft.com/en-us/azure/machine-learning/
- Azure ML Studio: https://ml.azure.com
- Check job logs for detailed error messages

---

**🎯 Focus:** Complete this Phase 1 in 4 hours, then you'll have a solid ML foundation to build upon!