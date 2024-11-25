
# Image Segmentation and Explainability Framework

This repository contains a comprehensive framework for processing image datasets, performing segmentation, training machine learning models, and generating explainability visualizations. It leverages state-of-the-art deep learning models and explainability techniques like **LIME** and **Grad-CAM**.

## Features

- **Dataset Preprocessing**
  - Handles multiple datasets, including rice leaf disease, breast ultrasound, and X-ray images.
  - Resizes, normalizes, and prepares images for training and evaluation.

- **Machine Learning Models**
  - Integrates pre-trained models such as:
    - EfficientNet
    - DenseNet121
    - VGG19
  - Customizable architectures for classification and explainability.

- **Image Segmentation**
  - Implements segmentation algorithms:
    - **SLIC (Simple Linear Iterative Clustering)**
    - **Felzenszwalb's Method**
    - **Quickshift**
    - **Watershed**
  - Highlights regions of interest for medical and agricultural applications.

- **Explainability**
  - **LIME**: Visualize the regions influencing model predictions.
  - **Grad-CAM**: Heatmaps to explain model decision-making.

- **Statistical Analysis**
  - Computes mean and standard deviation across datasets.
  - Calculates diseased leaf area percentage for agricultural applications.

- **Visualization**
  - Displays:
    - Class mean images.
    - Segmentation results.
    - Grad-CAM overlays.
    - Highlighted influential regions.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repository.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dependencies

- Python 3.7+
- TensorFlow
- OpenCV
- NumPy
- Matplotlib
- scikit-image
- LIME (Local Interpretable Model-Agnostic Explanations)

## Dataset

The framework supports multiple datasets, such as:
- [Rice Leaf Disease Dataset](https://example.com/dataset)
- [Breast Ultrasound Images Dataset](https://example.com/dataset)
- [Chest X-ray Pneumonia Dataset](https://example.com/dataset)

Ensure datasets are organized in folders corresponding to class labels.

## Usage

### 1. Dataset Preprocessing
Edit the `folder_path` variable in the script to point to your dataset directory. Run the following code to compute mean images:
```python
python slime_(1).py
```

### 2. Training a Model
Modify the model architecture and dataset paths as needed in the script. Example:
```python
model = build_model(dim=256, ef=0)
```

### 3. Explainability
Generate explainability visualizations using LIME:
```python
get_explanations(image_names=["image_name"], num_samples=100, random_state=0)
```

### 4. Segmentation
Run segmentation and visualizations for specific images:
```python
get_segmentation(image_names=["image_name"], segmentation_fn="SLIC")
```

## Results

- **Mean and Standard Deviation Analysis**:
  Displays statistical summaries for each dataset.
- **Segmentation Visualizations**:
  Highlight segmented regions in images using advanced methods.
- **Grad-CAM Heatmaps**:
  Heatmaps to visualize the model's decision-making process.

## Example Outputs

### Mean Class Images
![Mean Class Image](examples/mean_class_image.png)

### LIME Visualizations
![LIME Visualization](examples/lime_visualization.png)

### Grad-CAM Heatmaps
![Grad-CAM Heatmap](examples/grad_cam_heatmap.png)

## Contributions

Contributions are welcome! Please open a pull request or issue to discuss potential improvements or new features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
