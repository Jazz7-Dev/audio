# 🎵 Audio Classification Web Application

A powerful audio and video classification web application powered by **YAMNet** (Google's state-of-the-art audio event classifier). Upload any audio or video file to instantly identify sounds using deep learning.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## ✨ Features

- 🎤 **Audio Classification**: Classify 521 different sound categories
- 🎬 **Video Support**: Automatically extract and analyze audio from video files
- 🌐 **Web Interface**: Modern drag-and-drop file upload interface
- 🔌 **REST API**: JSON API endpoint for programmatic access
- 📊 **Transfer Learning**: Fine-tune YAMNet on custom datasets (ESC-50)
- 📈 **Training Visualization**: Generate professional training graphs

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Deep Learning | TensorFlow, TensorFlow Hub |
| Audio Model | [YAMNet](https://tfhub.dev/google/yamnet/1) (521 audio classes) |
| Audio Processing | Librosa, MoviePy |
| Web Framework | Flask |
| Training Dataset | [ESC-50](https://github.com/karolpiczak/ESC-50) |

## 📁 Project Structure

```
audio/
├── app.py              # Flask web application
├── sound.py            # Core audio classification logic
├── train_model.py      # YAMNet fine-tuning script
├── plot_training.py    # Training visualization
├── analyzer.py         # Audio extraction utilities
├── templates/
│   ├── index.html      # Upload page UI
│   └── result.html     # Results display
├── uploads/            # Uploaded files directory
└── yamnet_class_map.csv # YAMNet class labels
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Jazz7-Dev/audio.git
   cd audio
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install flask tensorflow tensorflow-hub librosa moviepy colorama numpy matplotlib
   ```

## 🎯 Usage

### Web Application

1. Start the Flask server:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to: `http://localhost:5000`

3. Drag and drop or select an audio/video file to analyze

### Command Line

```bash
python sound.py
```
Edit the `file_path` variable in `sound.py` to point to your audio file.

### API Endpoint

Send a POST request to `/analyze`:

```bash
curl -X POST -F "file=@your_audio.wav" http://localhost:5000/analyze
```

Response:
```json
{
  "results": [
    {"name": "Speech", "percentage": 45.23},
    {"name": "Music", "percentage": 22.15},
    ...
  ]
}
```

## 📊 Supported File Formats

| Audio | Video |
|-------|-------|
| WAV | MP4 |
| MP3 | MOV |
| FLAC | WebM |
| OGG | M4A |

## 🧠 Model Training (Optional)

Fine-tune YAMNet on the ESC-50 dataset:

```bash
python train_model.py
```

This will:
- Download ESC-50 dataset automatically
- Extract YAMNet embeddings
- Train a classifier on 50 environmental sound classes
- Save the model and training history

### Generate Training Graphs

```bash
python plot_training.py
```

Outputs professional graphs in `training_graphs/` directory.

## 📈 How It Works

1. **Audio Loading**: Supports multiple formats via Librosa; extracts audio from video using MoviePy
2. **Preprocessing**: Resamples audio to 16kHz (YAMNet requirement)
3. **Feature Extraction**: YAMNet generates 1024-dimensional embeddings
4. **Classification**: Maps embeddings to 521 AudioSet classes
5. **Aggregation**: Combines frame-level predictions into overall percentages

## 🔧 Configuration

Key settings in the code:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SAMPLE_RATE` | 16000 | Audio sample rate (Hz) |
| `top_n` | 15 | Number of top classes to display |
| `EPOCHS` | 20 | Training epochs (fine-tuning) |
| `BATCH_SIZE` | 32 | Training batch size |

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Devansh**  
GitHub: [@Jazz7-Dev](https://github.com/Jazz7-Dev)

## 🙏 Acknowledgments

- [YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet) by Google
- [ESC-50 Dataset](https://github.com/karolpiczak/ESC-50) by Karol Piczak
- TensorFlow Hub team

---

⭐ If you found this project helpful, please give it a star!
