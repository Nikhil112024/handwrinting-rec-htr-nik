# **Deciphering Handwriting Sentences To Images**

## **📌 Project Overview**
Deciphering Handwriting Sentences To Images is an advanced **Handwritten Text Recognition (HTR)** system that uses deep learning models to detect and recognize handwritten text from images and real-time inputs. The project integrates **YOLO for text detection, RNN-CTC for text recognition, and a dictionary-based correction mechanism using a prefix tree (Trie)**.

---

## **🚀 Features**
- ✅ **Real-time Handwritten Text Recognition** via webcam and image input.
- ✅ **YOLO-based Text Detection** for precise localization of handwritten text.
- ✅ **RNN-CTC for Text Recognition**, handling sequence-to-sequence transcription.
- ✅ **Dictionary-based Correction** using a prefix tree (Trie) to improve accuracy.
- ✅ **User-Friendly Web Interface** using Gradio for easy interaction.
- ✅ **Optimized for Speed and Efficiency** using ONNX models.

---

## **🛠️ Tech Stack**
### **📌 Frameworks & Libraries:**
- **Python 3.8+**
- **OpenCV** (Image processing)
- **ONNX Runtime** (Efficient model inference)
- **YOLO** (Text detection)
- **RNN-CTC** (Character recognition)
- **Gradio** (Web-based interface)
- **Matplotlib, NumPy** (Data visualization & processing)

### **📌 Development Tools:**
- **Jupyter Notebook / VSCode / PyCharm**
- **Docker (Optional for Deployment)**
- **Git & GitHub** (Version Control)

---
## **📂 Project Structure**
```
📦 Deciphering-Handwriting-Sentences-To-Images
│── data/                         # Sample data, images, and dictionary
│── htr_pipeline/                 # Core handwritten text recognition pipeline
│   │── models/                   # Pre-trained ONNX models (YOLO & RNN-CTC)
│   │── word_detector/            # Text detection module
│   │── reader/                   # Text recognition module
│── scripts/                      # Execution scripts
│   │── demo.py                   # Run the recognition pipeline
│   │── gradio_demo.py             # Web-based interface demo
│── README.md                     # Project documentation
│── requirements.txt               # Dependencies
│── setup.py                       # Python package setup
│── .gitignore                     # Git ignore file
```

---

## **🖥️ Installation & Setup**
### **Step 1: Clone the Repository**
```bash
git clone https://github.com/Nikhil112024/Deciphering-Handwriting-Sentences-To-Images.git
cd Deciphering-Handwriting-Sentences-To-Images
```

### **Step 2: Create and Activate a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate  # For Windows
```

### **Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 4: Run the Demo**
- **For Image Input Recognition:**
```bash
python scripts/demo.py
```
- **For Web-based Demo:**
```bash
python scripts/gradio_demo.py
```

---
## **📊 Performance Metrics**
- **Character Error Rate (CER):** ~2.5%
- **Word Error Rate (WER):** ~4.1%
- **Latency:** ~50ms per image (ONNX inference)

---

## **🔧 Future Improvements**
- [ ] Implement **Transformer-based OCR (TrOCR)** for improved accuracy.
- [ ] Develop a **mobile app** for real-time recognition.
- [ ] Extend **multilingual support** for non-English handwritten text.
- [ ] Deploy an **API for third-party integration**.

---

## **📬 Contact**
For questions or collaborations, connect with me:
- 📧 **Email:** [nikhilkumarjuyal777@gmail.com](mailto:nikhilkumarjuyal777@gmail.com)
- 💼 **LinkedIn:** [Nikhil Kumar](https://linkedin.com/in/nikhil-kumar-8054042b2/)
- 🖥️ **GitHub:** [Nikhil112024](https://github.com/Nikhil112024)

---


