# Quick Start Guide

## Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- CUDA-capable GPU (optional but recommended)
- 8GB+ RAM

## Backend Setup

### 1. Navigate to backend directory
```bash
cd backend
```

### 2. Create and activate virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Create environment file
```bash
copy .env.example .env
```

Edit `.env` and configure settings as needed.

### 5. Start the backend server
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`
API documentation: `http://localhost:8000/api/docs`

## Frontend Setup

### 1. Navigate to frontend directory
```bash
cd frontend
```

### 2. Install dependencies
```bash
npm install
```

### 3. Start development server
```bash
npm run dev
```

The application will be available at `http://localhost:3000`

## Using the Application

1. **Image Detection**: Upload a single image to analyze
2. **Video Detection**: Upload a video and configure frame sampling
3. **Batch Detection**: Upload multiple files for batch analysis

## Notes

- **Model Files**: The system will work without pre-trained models (demo mode). For production use, train models using the ML pipeline.
- **GPU Support**: If CUDA is not available, the system will automatically fall back to CPU.
- **File Limits**: Maximum file size is 100MB. Batch processing supports up to 20 files.

## Troubleshooting

### Backend Issues

**Import errors:**
```bash
pip install --upgrade -r requirements.txt
```

**CUDA not available:**
- Edit `.env` and set `DEVICE=cpu`

### Frontend Issues

**Port already in use:**
```bash
# Edit vite.config.js and change the port
```

**API connection errors:**
- Ensure backend is running on port 8000
- Check CORS settings in backend config

## Next Steps

- Train models on your own dataset
- Deploy to production
- Customize UI theme
- Add additional features

For detailed documentation, see the main README.md file.
