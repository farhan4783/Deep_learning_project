# 🚀 DeepGuard - Quick Start Guide

## Prerequisites

- **Python**: 3.8 or higher
- **Node.js**: 16 or higher  
- **GPU**: NVIDIA GPU with CUDA 11.8+ (recommended, but CPU works too)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB for models and dependencies

---

## Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd deepfake-detection
```

### 2. Backend Setup

```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup

```bash
# Navigate to frontend (from project root)
cd frontend

# Install dependencies
npm install
```

---

## Running the Application

### Start Backend Server

```bash
# From backend directory with venv activated
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend API will be available at: `http://localhost:8000`

### Start Frontend Development Server

```bash
# From frontend directory
npm run dev
```

The frontend will be available at: `http://localhost:5173`

---

## First-Time Setup

### Download Pre-trained Models (Optional)

If you have pre-trained model weights:

```bash
# Create models directory
mkdir -p backend/models

# Place your model files in backend/models/
# - efficientnet_b4.pth
# - xception.pth
# - vit_base.pth
# etc.
```

### Environment Configuration

Create `.env` file in `backend/` directory:

```env
# Server Configuration
HOST=0.0.0.0
PORT=8000

# Model Configuration
MODEL_PATH=./models
DEVICE=cuda  # or 'cpu' if no GPU

# WebSocket Configuration
WS_MAX_CONNECTIONS=100
WS_PING_INTERVAL=30

# Processing Configuration
CHUNK_SIZE=16
FRAME_SKIP=1
MAX_VIDEO_SIZE=524288000  # 500MB in bytes
```

---

## Testing the Application

### 1. Open Browser

Navigate to `http://localhost:5173`

### 2. Upload an Image

- Click on "Image Detection" page
- Drag and drop an image or click to browse
- Wait for detection results
- View confidence scores and heatmap explanations

### 3. Upload a Video

- Click on "Video Detection" page
- Upload a video file (MP4, AVI, MOV)
- Watch real-time progress updates
- Explore frame-by-frame results on the interactive timeline

### 4. Batch Processing

- Click on "Batch Detection" page
- Upload multiple files
- View aggregated results

---

## Advanced Features

### Real-time Updates

The application uses WebSocket for real-time progress updates:
- Upload progress
- Frame-by-frame detection results
- Processing stage indicators
- Final results delivery

### 3D Visualization

View model confidence scores in an interactive 3D bar chart:
- Rotate and zoom with mouse
- Hover for detailed tooltips
- Color-coded confidence levels

### Video Timeline

Scrub through video analysis results:
- Click timeline to jump to frames
- View confidence graph overlay
- Play/pause through frames
- Export timeline data

---

## Troubleshooting

### Backend Issues

**Problem**: `ModuleNotFoundError` for torch or other packages

**Solution**:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Problem**: CUDA out of memory

**Solution**:
- Reduce batch size in config
- Use CPU instead: set `DEVICE=cpu` in `.env`
- Process smaller videos

### Frontend Issues

**Problem**: `npm install` fails

**Solution**:
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and package-lock.json
rm -rf node_modules package-lock.json

# Reinstall
npm install
```

**Problem**: WebSocket connection fails

**Solution**:
- Ensure backend is running on port 8000
- Check firewall settings
- Verify `vite.config.js` proxy configuration

### Performance Issues

**Problem**: Slow detection

**Solutions**:
- Enable GPU: Install CUDA and set `DEVICE=cuda`
- Reduce video resolution
- Increase `FRAME_SKIP` to process fewer frames
- Use smaller models (Custom CNN instead of ViT)

---

## Development Tips

### Hot Reload

Both frontend and backend support hot reload:
- **Backend**: `--reload` flag automatically reloads on code changes
- **Frontend**: Vite automatically reloads on file changes

### API Documentation

Access interactive API docs at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### WebSocket Testing

Test WebSocket connection:
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/test-client');
ws.onmessage = (event) => console.log(JSON.parse(event.data));
ws.send(JSON.stringify({ type: 'ping' }));
```

---

## Production Deployment

### Using Docker (Recommended)

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access application
# Frontend: http://localhost:3000
# Backend: http://localhost:8000
```

### Manual Deployment

**Backend**:
```bash
# Install production server
pip install gunicorn

# Run with gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

**Frontend**:
```bash
# Build for production
npm run build

# Serve with nginx or any static server
# Build output is in dist/
```

---

## Next Steps

1. **Train Models**: Use your own dataset to train custom models
2. **Customize UI**: Modify frontend components to match your brand
3. **Add Features**: Extend with user authentication, analytics, etc.
4. **Deploy**: Set up production environment with Docker/Kubernetes
5. **Monitor**: Add logging and monitoring for production use

---

## Resources

- **Documentation**: See `README.md` for full documentation
- **Walkthrough**: Check `walkthrough.md` for implementation details
- **API Reference**: Visit `/docs` endpoint for API documentation
- **Issues**: Report bugs on GitHub Issues

---

## Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the full README and walkthrough
3. Open an issue on GitHub
4. Contact the maintainers

---

**Happy Detecting! 🎭**
