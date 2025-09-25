# P&ID PDF Processor with YOLO Detection

An intelligent PDF processing application that uses computer vision and OCR to automatically detect, classify, and extract text from Piping & Instrumentation Diagrams (P&ID). Built with a custom-trained YOLO model for shape detection and multiple OCR engines for text extraction.

## 🚀 Features

- **Custom YOLO Model**: Trained specifically for P&ID component detection
- **Multi-OCR Support**: Choose between PaddleOCR, EasyOCR, or custom OCR pipelines
- **Interactive GUI**: Drag-and-drop interface built with Tkinter
- **Batch Processing**: Process multiple PDF pages efficiently
- **Export Functionality**: Export results to Excel (XLSX) format
- **Visual Detection**: View detected components with bounding boxes and confidence scores
- **Robust Error Handling**: Multiple fallback strategies for reliable processing

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- Windows/Linux/macOS
- Minimum 8GB RAM (16GB recommended for large PDFs)
- CUDA-compatible GPU (optional, for faster processing)

### Dependencies
```
ultralytics>=8.0.0
opencv-python>=4.5.0
pandas>=1.3.0
matplotlib>=3.3.0
pdf2image>=2.1.0
pillow>=8.0.0
numpy>=1.21.0
tkinter (usually included with Python)
tkinterdnd2>=0.3.0
easyocr>=1.6.0
paddlepaddle>=2.4.0
paddlex>=2.1.0
```

### Additional Requirements
- **Poppler**: Required for PDF processing
  - Windows: Download from [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases)
  - Ubuntu/Debian: `sudo apt-get install poppler-utils`
  - macOS: `brew install poppler`

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/pid-pdf-processor.git
cd pid-pdf-processor
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download and setup Poppler**
- Update the `poppler_path` variable in `PDFProcessor.py` to match your installation

5. **Place your trained YOLO model**
- Put your `best.pt` model file in the `models/` directory
- Or update the model path in `get_model()` function

## 📖 Usage

### GUI Application
```bash
python main.py
```

1. **Load PDF**: Drag and drop a PDF file or use the browse button
2. **Processing**: The application will automatically detect P&ID components
3. **Review Results**: View detected shapes and extracted text in the data grid
4. **Export**: Save results to Excel format

### Programmatic Usage
```python
from PDFProcessor import get_data_from_pdf_easyocr

# Process PDF with EasyOCR
df = get_data_from_pdf_easyocr(
    pdf_path="your_pid_diagram.pdf",
    progress_callback=None,
    visualize='matplotlib'
)

# Export results
df.to_excel("results.xlsx", index=False)
```

## 🎯 YOLO Model Training

### Dataset Preparation
1. **Collect P&ID Images**: Gather diverse P&ID diagrams
2. **Annotation**: Use tools like [LabelImg](https://github.com/heartexlabs/labelImg) or [Roboflow](https://roboflow.com)
3. **Classes**: Define your P&ID component classes (e.g., valves, pumps, instruments, pipes)

### Training Process
```bash
# Install Ultralytics
pip install ultralytics

# Train the model
yolo train data=pid_dataset.yaml model=yolov8n.pt epochs=100 imgsz=640

# Validate the model
yolo val model=runs/detect/train/weights/best.pt data=pid_dataset.yaml

# Run inference
yolo predict model=runs/detect/train/weights/best.pt source=test_images/
```

### Dataset Structure
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── pid_dataset.yaml
```

### Sample `pid_dataset.yaml`
```yaml
path: ./dataset
train: images/train
val: images/val
test: images/test

nc: 8  # number of classes
names: ['valve', 'pump', 'instrument', 'pipe', 'tank', 'heat_exchanger', 'compressor', 'control_valve']
```

## 🔧 Configuration

### OCR Engine Selection
Choose your preferred OCR engine by calling the appropriate function:

- **EasyOCR** (Recommended): `get_data_from_pdf_easyocr()`
- **PaddleOCR**: `get_data_from_pdf()`
- **Memory-based**: `get_data_from_pdf_memory()`

### Visualization Options
```python
# OpenCV visualization
get_data_from_pdf_easyocr(visualize='cv2')

# Matplotlib visualization (default)
get_data_from_pdf_easyocr(visualize='matplotlib')

# No visualization
get_data_from_pdf_easyocr(visualize=None)
```

## 📊 Output Format

The application generates a structured DataFrame with the following columns:

| Column | Description |
|--------|-------------|
| Shape | Detected P&ID component type |
| Label | Extracted text from OCR |
| X, Y | Top-left coordinates of bounding box |
| Width, Height | Dimensions of detected component |
| PDF Name | Source PDF filename |

## 🐛 Troubleshooting

### Common Issues

1. **Model not found**
   - Ensure `best.pt` is in the correct directory
   - Check file permissions

2. **Poppler not found**
   - Verify Poppler installation
   - Update `poppler_path` in configuration

3. **OCR failures**
   - Try different OCR engines
   - Check image quality and resolution
   - Ensure sufficient memory is available

4. **Memory issues**
   - Reduce batch size
   - Use sequential processing instead of parallel
   - Close other applications to free up RAM

### Performance Optimization

- **GPU Acceleration**: Ensure CUDA is properly installed for faster inference
- **Image Preprocessing**: Adjust DPI and image enhancement parameters
- **Model Optimization**: Consider using YOLOv8s or YOLOv8m for better accuracy vs speed trade-offs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black .
isort .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for text recognition
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for alternative OCR capabilities
- [pdf2image](https://github.com/Belval/pdf2image) for PDF processing

## 📞 Support

For questions, issues, or feature requests, please:
1. Check the [Issues](https://github.com/yourusername/pid-pdf-processor/issues) page
2. Create a new issue with detailed information
3. Contact: rhrj@ramboll.com

## 🚀 Roadmap

- [ ] Support for multi-page PDF processing
- [ ] Advanced P&ID component relationship mapping
- [ ] Integration with CAD software APIs
- [ ] Real-time processing capabilities
- [ ] Web-based interface option
- [ ] Docker containerization
- [ ] Cloud deployment options

---

**Made with ❤️ for the Process Engineering Community**
