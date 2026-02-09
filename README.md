# 🎭 DeepGuard - AI-Powered Deepfake Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green)
![React](https://img.shields.io/badge/React-18.0%2B-cyan)

Know Lets start with deeplearning through this project.
A state-of-the-art deep learning system for detecting deepfake images and videos using ensemble neural networks with explainable AI features.

## 🌟 Features

### Core Capabilities
- **Multi-Model Ensemble**: Combines EfficientNet, XceptionNet, and custom CNN architectures
- **Image & Video Detection**: Supports both static images and video frame analysis
- **Explainable AI**: Grad-CAM visualization showing which regions influenced the prediction
- **Real-time Processing**: Fast inference with optimized model architecture
- **Batch Processing**: Handle multiple files simultaneously

### Advanced Features
- **Attention Mechanisms**: Enhanced feature extraction with self-attention layers
- **Temporal Analysis**: Video-specific features for detecting temporal inconsistencies
- **Confidence Scoring**: Detailed probability scores with uncertainty estimation
- **Face Detection Integration**: Automatic face extraction and analysis
- **Data Augmentation**: Robust training with advanced augmentation techniques

## 🏗️ Architecture

```
DeepGuard/
├── backend/                 # FastAPI server
│   ├── app/
│   │   ├── api/            # API endpoints
│   │   ├── models/         # ML model definitions
│   │   ├── services/       # Business logic
│   │   └── utils/          # Helper functions
│   └── requirements.txt
├── frontend/               # React application
│   ├── src/
│   │   ├── components/    # UI components
│   │   ├── pages/         # Page components
│   │   ├── services/      # API integration
│   │   └── styles/        # CSS modules
│   └── package.json
├── ml/                     # Machine learning pipeline
│   ├── data/              # Dataset utilities
│   ├── models/            # Model architectures
│   ├── training/          # Training scripts
│   └── evaluation/        # Evaluation metrics
└── notebooks/             # Jupyter notebooks
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models (optional)
python scripts/download_models.py

# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### Training Your Own Model

```bash
# Navigate to ML directory
cd ml

# Download dataset (FaceForensics++, Celeb-DF, etc.)
python data/download_dataset.py --dataset faceforensics

# Preprocess data
python data/preprocess.py --input data/raw --output data/processed

# Train model
python training/train.py --config configs/efficientnet_b4.yaml

# Evaluate model
python evaluation/evaluate.py --model checkpoints/best_model.pth
```

## 📊 Model Performance

| Model | Accuracy | AUC-ROC | F1-Score | Inference Time |
|-------|----------|---------|----------|----------------|
| EfficientNet-B4 | 96.8% | 0.989 | 0.967 | 45ms |
| XceptionNet | 95.2% | 0.982 | 0.951 | 38ms |
| Custom CNN | 93.5% | 0.971 | 0.934 | 22ms |
| **Ensemble** | **98.1%** | **0.994** | **0.981** | 105ms |

*Tested on FaceForensics++ and Celeb-DF datasets*

## 🎯 API Endpoints

### Image Detection
```http
POST /api/v1/detect/image
Content-Type: multipart/form-data

{
  "file": <image_file>,
  "explain": true
}
```

### Video Detection
```http
POST /api/v1/detect/video
Content-Type: multipart/form-data

{
  "file": <video_file>,
  "frame_sampling": "uniform",
  "num_frames": 32
}
```

### Batch Processing
```http
POST /api/v1/detect/batch
Content-Type: multipart/form-data

{
  "files": [<file1>, <file2>, ...],
  "explain": false
}
```

## 🧠 Model Details

### EfficientNet-B4 Detector
- Pre-trained on ImageNet
- Fine-tuned on deepfake datasets
- Input size: 380x380
- Parameters: ~19M

### XceptionNet Detector
- Modified Xception architecture
- Optimized for face manipulation detection
- Input size: 299x299
- Parameters: ~23M

### Custom CNN with Attention
- Lightweight architecture
- Self-attention layers
- Input size: 224x224
- Parameters: ~8M

## 📈 Training Data

The system is trained on multiple datasets:
- **FaceForensics++**: 1000 original + 4000 manipulated videos
- **Celeb-DF**: 590 real + 5639 deepfake videos
- **DFDC**: 100,000+ videos from Facebook challenge
- **Custom Dataset**: Additional curated examples

## 🎨 Frontend Features

- **Modern Glassmorphism UI**: Premium design with smooth animations
- **Drag & Drop Upload**: Intuitive file upload interface
- **Real-time Progress**: Live processing status updates
- **Heatmap Visualization**: Interactive Grad-CAM overlays
- **Batch Results Dashboard**: Comprehensive results view
- **Dark/Light Theme**: User preference support
- **Responsive Design**: Mobile and desktop optimized

## 🔧 Configuration

### Model Configuration (`ml/configs/config.yaml`)
```yaml
model:
  architecture: efficientnet_b4
  pretrained: true
  num_classes: 2
  dropout: 0.3

training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.0001
  optimizer: adam
  scheduler: cosine

data:
  image_size: 380
  augmentation: true
  normalization: imagenet
```

## 🧪 Testing

```bash
# Run backend tests
cd backend
pytest tests/ -v

# Run frontend tests
cd frontend
npm test

# Run ML pipeline tests
cd ml
python -m pytest tests/
```

## 📦 Deployment

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up -d

# Access application
# Frontend: http://localhost:3000
# Backend: http://localhost:8000
```

### Cloud Deployment
- AWS: Use EC2 with GPU instances (p3.2xlarge recommended)
- Google Cloud: Use Compute Engine with T4 GPUs
- Azure: Use NC-series VMs

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- FaceForensics++ dataset creators
- PyTorch and FastAPI communities
- Research papers on deepfake detection
- Open-source contributors

## 📚 References

1. Rossler et al. (2019) - FaceForensics++
2. Li et al. (2020) - Celeb-DF
3. Tan & Le (2019) - EfficientNet
4. Chollet (2017) - Xception

## 📧 Contact

For questions or support, please open an issue or contact the maintainers.

---

**⚠️ Disclaimer**: This tool is for research and educational purposes. Always verify important content through multiple sources.
