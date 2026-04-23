# 🌿 Deforestation Detection

A Digital Image Processing (DIP) project that detects and quantifies deforestation by comparing before/after satellite or aerial images using classical computer vision techniques.

---

## 📌 About the Project

This project implements a **change detection pipeline** to identify deforested regions between two time-period images of the same area. It uses a series of image processing steps — normalization, Gaussian blurring, pixel-difference computation, thresholding, and contour detection — to highlight areas where forest cover has been lost, and estimates a deforestation percentage.

The results are presented through a clean, interactive **Streamlit web app** with a dark forest-themed UI.

---

## 🧠 How It Works

The pipeline processes a pair of **Before** and **After** images through the following stages:

```
Input Images → Normalize → Gaussian Blur → Grayscale → Difference Map → Threshold → Contour Detection → Classification
```

| Step | Description |
|------|-------------|
| **Normalization** | Min-max normalizes pixel values to `[0, 255]` for consistent comparison |
| **Gaussian Blur** | Applies a 5×5 Gaussian filter (σ=1.0) to reduce noise |
| **Grayscale Conversion** | Converts to single-channel for pixel-wise subtraction |
| **Difference Map** | Computes absolute pixel difference between before and after |
| **Thresholding** | Creates a binary change mask using the 82nd-percentile threshold |
| **Contour Detection** | Detects and filters contours of changed regions (min area: 40px) |
| **Classification** | Classifies output as *Deforestation Detected*, *Reforestation Detected*, or *No Significant Change* |

---

## 📊 Output Metrics

- **Deforestation %** — Estimated percentage of forest lost
- **Area Changed %** — Percentage of total image area that changed
- **Forest Loss Status** — Severity: `Low` / `Moderate` / `Severe` / `Critical`
- **Final Classification** — One of three output labels

---

## 🖥️ App Features

- Upload **Before/After** image pairs manually, or load from a structured `dataset/` folder
- Step-by-step visual output of every pipeline stage
- Contour-highlighted overlay showing changed regions
- Side-by-side before/after comparison view
- Final summary with all metrics

---

## 📁 Project Structure

```
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── SarthakSabharwal_2401020234_Deforestation_Detection.ipynb   # Jupyter notebook
└── dataset/                # Image dataset (not included in repo)
    ├── deforestation/      # Post-deforestation images
    └── no_deforestation/   # Pre-deforestation (baseline) images
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/deforestation-detection.git
cd deforestation-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

### 4. Add your dataset *(optional)*

Place paired images in the `dataset/` folder:

```
dataset/
  deforestation/   ← "after" images (post-deforestation)
  no_deforestation/ ← "before" images (baseline/healthy forest)
```

Or simply use the **Upload manually** option in the app to upload any two images directly.

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `Streamlit` | Web UI framework |
| `OpenCV` | Image processing (blur, contours, thresholding) |
| `NumPy` | Array operations and math |
| `Pillow` | Image loading and format handling |
| `Matplotlib` | Difference map heatmap visualization |

---

## 📓 Notebook

The Jupyter notebook (`SarthakSabharwal_2401020234_Deforestation_Detection.ipynb`) contains the full experimental walkthrough of the pipeline with inline visualizations, suitable for academic submission.

---

## 📄 License

This project is for academic/educational purposes as part of a Digital Image Processing course.
