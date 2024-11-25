img_list = ['Rice Blast_0009']
plt.rcParams['figure.figsize'] = [18, 5 * len(img_list)]
get_explanations(img_list, num_samples=DEFAULT_NUM_SAMPLES, random_state=0)

import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

# Define the path to the specific image
image_path = '/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/person1001_bacteria_2932.jpeg'

# Load the image
img = cv2.imread(image_path, cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

# Assuming IMAGE_SIZE is the size required by your model
IMAGE_SIZE = [112, 112]  # Replace with your model's input size

# Resize and normalize the image
image_resized = cv2.resize(img, (IMAGE_SIZE[0], IMAGE_SIZE[1]))
image_normalized = image_resized / 255.0

# Convert to tensor
image_tensor = tf.convert_to_tensor(image_normalized, dtype=tf.float32)
image_tensor = tf.expand_dims(image_tensor, 0)  # Add batch dimension

# Display the image
plt.imshow(image_normalized)
plt.axis('off')
plt.show()

import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

# Define the path to the specific image
image_path = '/kaggle/input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT/malignant/malignant (1).png'

# Load the image
img = cv2.imread(image_path, cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

# Assuming IMAGE_SIZE is the size required by your model
IMAGE_SIZE = [112, 112]  # Replace with your model's input size

# Resize and normalize the image
image_resized = cv2.resize(img, (IMAGE_SIZE[0], IMAGE_SIZE[1]))
image_normalized = image_resized / 255.0

# Convert to tensor
image_tensor = tf.convert_to_tensor(image_normalized, dtype=tf.float32)
image_tensor = tf.expand_dims(image_tensor, 0)  # Add batch dimension

# Display the image
plt.imshow(image_normalized)
plt.axis('off')
plt.show()

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

segments_fz = felzenszwalb(img, scale=200, sigma=0.5, min_size=50)
segments_slic = slic(img, n_segments=50, compactness=10, sigma=1)
segments_quick = quickshift(img, kernel_size=2, max_dist=100, ratio=0.5)
gradient = sobel(rgb2gray(img))
segments_watershed = watershed(gradient, markers=25, compactness=0.001)

print(f"Felzenszwalb number of segments: {len(np.unique(segments_fz))}")
print(f"SLIC number of segments: {len(np.unique(segments_slic))}")
print(f"Quickshift number of segments: {len(np.unique(segments_quick))}")
print(f"Watershed number of segments: {len(np.unique(segments_watershed))}")
fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

ax[0, 0].imshow(mark_boundaries(img, segments_fz))
ax[0, 0].set_title("Felzenszwalbs's method")
ax[0, 1].imshow(mark_boundaries(img, segments_slic))
ax[0, 1].set_title('SLIC')
ax[1, 0].imshow(mark_boundaries(img, segments_quick))
ax[1, 0].set_title('Quickshift')
ax[1, 1].imshow(mark_boundaries(img, segments_watershed))
ax[1, 1].set_title('Compact watershed')

for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()

from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.metrics import boundary_precision_recall
import numpy as np
import matplotlib.pyplot as plt

# Example image (replace with your own)
img = img_as_float(plt.imread('/kaggle/input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT/malignant/malignant (10).png'))  # Replace 'your_image.png' with your actual image path

# Segment the image using different algorithms
segments_fz = felzenszwalb(img, scale=200, sigma=0.5, min_size=50)
segments_slic = slic(img, n_segments=50, compactness=10, sigma=1)
segments_quick = quickshift(img, kernel_size=2, max_dist=100, ratio=0.5)
gradient = sobel(rgb2gray(img))
segments_watershed = watershed(gradient, markers=25, compactness=0.001)

# Ground truth segmentation (replace with your ground truth if available)
ground_truth = np.zeros(img.shape[:2])  # Replace this with the actual ground truth mask

# Compute Boundary Precision (BP) and Boundary Recall (BR)
bp_fz, br_fz = boundary_precision_recall(segments_fz, ground_truth)
bp_slic, br_slic = boundary_precision_recall(segments_slic, ground_truth)
bp_quick, br_quick = boundary_precision_recall(segments_quick, ground_truth)
bp_watershed, br_watershed = boundary_precision_recall(segments_watershed, ground_truth)

# Print number of segments and BP/BR for each method
print(f"Felzenszwalb number of segments: {len(np.unique(segments_fz))}, BP: {bp_fz:.4f}, BR: {br_fz:.4f}")
print(f"SLIC number of segments: {len(np.unique(segments_slic))}, BP: {bp_slic:.4f}, BR: {br_slic:.4f}")
print(f"Quickshift number of segments: {len(np.unique(segments_quick))}, BP: {bp_quick:.4f}, BR: {br_quick:.4f}")
print(f"Watershed number of segments: {len(np.unique(segments_watershed))}, BP: {bp_watershed:.4f}, BR: {br_watershed:.4f}")

# Plot the segmentation results
fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

ax[0, 0].imshow(mark_boundaries(img, segments_fz))
ax[0, 0].set_title("Felzenszwalb's method")
ax[0, 1].imshow(mark_boundaries(img, segments_slic))
ax[0, 1].set_title('SLIC')
ax[1, 0].imshow(mark_boundaries(img, segments_quick))
ax[1, 0].set_title('Quickshift')
ax[1, 1].imshow(mark_boundaries(img, segments_watershed))
ax[1, 1].set_title('Compact watershed')

for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()





from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

segments_fz = felzenszwalb(img, scale=200, sigma=0.5, min_size=50)
segments_slic = slic(img, n_segments=50, compactness=10, sigma=1)
segments_quick = quickshift(img, kernel_size=2, max_dist=100, ratio=0.5)
gradient = sobel(rgb2gray(img))
segments_watershed = watershed(gradient, markers=25, compactness=0.001)

print(f"Felzenszwalb number of segments: {len(np.unique(segments_fz))}")
print(f"SLIC number of segments: {len(np.unique(segments_slic))}")
print(f"Quickshift number of segments: {len(np.unique(segments_quick))}")
print(f"Watershed number of segments: {len(np.unique(segments_watershed))}")
fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

ax[0, 0].imshow(mark_boundaries(img, segments_fz))
ax[0, 0].set_title("Felzenszwalbs's method")
ax[0, 1].imshow(mark_boundaries(img, segments_slic))
ax[0, 1].set_title('SLIC')
ax[1, 0].imshow(mark_boundaries(img, segments_quick))
ax[1, 0].set_title('Quickshift')
ax[1, 1].imshow(mark_boundaries(img, segments_watershed))
ax[1, 1].set_title('Compact watershed')

for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()

def get_explanations(image_names, num_samples=DEFAULT_NUM_SAMPLES, random_state=0):
    n_img = len(image_names) * 4  # 4 rows per image for each segmentation technique
    id_img = 1

    # Define segmentation functions in a list with their names
    segmentation_fns = [
        ('SLIC', lambda image: slic(image, n_segments=50, compactness=10, sigma=1)),
        ('Felzenszwalb', lambda image: felzenszwalb(image, scale=100, sigma=0.5, min_size=50)),
        ('Quickshift', lambda image: quickshift(image, kernel_size=3, max_dist=6, ratio=0.5)),
        ('Watershed', lambda image: watershed(sobel(rgb2gray(image)), markers=250, compactness=0.001))
    ]

    for image_name in image_names:
        explainer = lime_image.LimeImageExplainer(random_state=random_state)
        class_name = image_name.rsplit('_', 1)[0]
        file_path = f'/kaggle/input/chest-xray-pneumonia/{class_name}/{image_name}.jpg'

        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (IMAGE_SIZE[0], IMAGE_SIZE[1]))
        img = img / 255.0

        prob = model.predict(tf.expand_dims(img, axis=0))
        prd = np.argmax(prob, axis=-1)

        for seg_name, segment_fn in segmentation_fns:
            # Apply segmentation
            segments = segment_fn(img)
            num_segments = len(np.unique(segments))

            # Visualization and Lime explanations
            plt.subplot(n_img, 4, id_img)
            plt.title(f'Original - {seg_name} (Pred = {prd})')
            plt.imshow(img)
            id_img += 1

            plt.subplot(n_img, 4, id_img)
            plt.title(f'{seg_name} Segmentation ({num_segments} segments)')
            plt.imshow(mark_boundaries(img, segments))
            id_img += 1

            explanation = explainer.explain_instance(img, model.predict, num_samples=num_samples, segmentation_fn=segment_fn)
            temp, mask = explanation.get_image_and_mask(prd[0], positive_only=False, num_features=5, hide_rest=False, min_weight=0.0)
            plt.subplot(n_img, 4, id_img)
            plt.title(f'{seg_name} Pos/Neg Regions')
            plt.imshow(mark_boundaries(temp, mask))
            id_img += 1

            temp, mask = explanation.get_image_and_mask(prd[0], positive_only=True if prd[0] == 1 else False, negative_only=True if prd[0] == 0 else False, num_features=1, hide_rest=False, min_weight=0.0)
            plt.subplot(n_img, 4, id_img)
            plt.title(f'{seg_name} Top Negative Region')
            plt.imshow(mark_boundaries(temp, mask))
            id_img += 1

# Usage
img_list = ['person1002_bacteria_2933']
plt.rcParams['figure.figsize'] = [18, 20 * len(img_list)]
get_explanations(img_list, num_samples=DEFAULT_NUM_SAMPLES, random_state=0)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed

def segment_fn(image):
    return slic(image, n_segments=50, compactness=10, sigma=1)

def get_explanations(image_names, num_samples=DEFAULT_NUM_SAMPLES, random_state=0):
    n_img = len(image_names) * 4  # 4 rows per image for each segmentation technique
    id_img = 1

    # Define segmentation functions in a list with their names
    segmentation_fns = [
        ('SLIC', lambda image: slic(image, n_segments=50, compactness=10, sigma=1)),
        ('Felzenszwalb', lambda image: felzenszwalb(image, scale=100, sigma=0.5, min_size=50)),
        ('Quickshift', lambda image: quickshift(image, kernel_size=3, max_dist=6, ratio=0.5)),
        ('Watershed', lambda image: watershed(sobel(rgb2gray(image)), markers=250, compactness=0.001))
    ]

    for image_name in image_names:
        explainer = lime_image.LimeImageExplainer(random_state=random_state)
        class_name = image_name.rsplit('_', 1)[0]
        file_path = f'/kaggle/input/chest-xray-pneumonia/{class_name}/{image_name}.jpeg'

        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Unable to read the file at path: {file_path}")

        # Resize the image only if it's loaded successfully
        img = cv2.resize(img, (IMAGE_SIZE[0], IMAGE_SIZE[1]))
        img = img / 255.0

        prob = model.predict(tf.expand_dims(img, axis=0))
        prd = np.argmax(prob, axis=-1)

        for seg_name, segment_fn in segmentation_fns:
            # Apply segmentation
            segments = segment_fn(img)
            num_segments = len(np.unique(segments))

            # Visualization and Lime explanations
            plt.subplot(n_img, 4, id_img)
            plt.title(f'Original - {seg_name} (Pred = {prd})')
            plt.imshow(img)
            id_img += 1

            plt.subplot(n_img, 4, id_img)
            plt.title(f'{seg_name} Segmentation ({num_segments} segments)')
            plt.imshow(mark_boundaries(img, segments))
            id_img += 1

            explanation = explainer.explain_instance(img, model.predict, num_samples=num_samples, segmentation_fn=segment_fn)
            temp, mask = explanation.get_image_and_mask(prd[0], positive_only=False, num_features=5, hide_rest=False, min_weight=0.0)
            plt.subplot(n_img, 4, id_img)
            plt.title(f'{seg_name} Pos/Neg Regions')
            plt.imshow(mark_boundaries(temp, mask))
            id_img += 1

            temp, mask = explanation.get_image_and_mask(prd[0], positive_only=True if prd[0] == 1 else False, negative_only=True if prd[0] == 0 else False, num_features=1, hide_rest=False, min_weight=0.0)
            plt.subplot(n_img, 4, id_img)
            plt.title(f'{seg_name} Top Negative Region')
            plt.imshow(mark_boundaries(temp, mask))
            id_img += 1

# Usage
img_list = ['person1003_bacteria_2934']
plt.rcParams['figure.figsize'] = [18, 20 * len(img_list)]
get_explanations(img_list, num_samples=DEFAULT_NUM_SAMPLES, random_state=0)



def get_explanations(image_names, num_samples=DEFAULT_NUM_SAMPLES, random_state=0):
    for image_name in image_names:
        explainer = lime_image.LimeImageExplainer(random_state=random_state)
        class_name = image_name.rsplit('_', 1)[0]
        file_path = f'/kaggle/input/bd-rice-leaf-dataset/Noised Field Background/{class_name}/{image_name}.jpg'

        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = cv2.resize(img, (IMAGE_SIZE[0], IMAGE_SIZE[1]))
        img = img / 255.0

        prob = model.predict(tf.expand_dims(img, axis=0))
        prd = np.argmax(prob, axis=-1)

        # SLIC segmentation
        segments = slic(img, n_segments=50, compactness=10, sigma=1)
        explanation = explainer.explain_instance(img, model.predict, num_samples=num_samples, segmentation_fn=lambda x: segments)
        temp, mask = explanation.get_image_and_mask(prd[0], positive_only=True, num_features=5, hide_rest=False, min_weight=0.01)

        # Create a mask overlay for diseased segments
        mask_overlay = np.zeros_like(img)
        for segment_id in np.unique(segments):
            if mask[segments == segment_id].any():
                mask_overlay[segments == segment_id] = [1, 0, 0]  # Red color

        # Apply the mask overlay
        labeled_img = img.copy()
        labeled_img[mask_overlay.sum(axis=-1) > 0] = labeled_img[mask_overlay.sum(axis=-1) > 0] * 0.5 + mask_overlay[mask_overlay.sum(axis=-1) > 0] * 0.5

        # Visualization
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        plt.title('Original')
        plt.imshow(img)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title('Segmentation')
        plt.imshow(mark_boundaries(img, segments))
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title('Diseased Segments')
        plt.imshow(labeled_img)
        plt.axis('off')

        plt.show()

# Usage
img_list = ['Browon Spot_0001']
get_explanations(img_list, num_samples=DEFAULT_NUM_SAMPLES, random_state=0)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage import color

def get_explanations(image_names, num_samples=DEFAULT_NUM_SAMPLES, random_state=0):
    for image_name in image_names:
        explainer = lime_image.LimeImageExplainer(random_state=random_state)
        class_name = image_name.rsplit('_', 1)[0]
        file_path = f'/kaggle/input/bd-rice-leaf-dataset/Noised Field Background/{class_name}/{image_name}.jpg'

        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = cv2.resize(img, (IMAGE_SIZE[0], IMAGE_SIZE[1]))
        img = img / 255.0

        prob = model.predict(tf.expand_dims(img, axis=0))
        prd = np.argmax(prob, axis=-1)

        # SLIC segmentation
        segments = slic(img, n_segments=50, compactness=10, sigma=1)
        explanation = explainer.explain_instance(img, model.predict, num_samples=num_samples, segmentation_fn=lambda x: segments)
        temp, mask = explanation.get_image_and_mask(prd[0], positive_only=True, num_features=5, hide_rest=False, min_weight=0.01)

        # Create a mask overlay for diseased segments
        mask_overlay = np.zeros_like(img)
        for segment_id in np.unique(segments):
            if mask[segments == segment_id].any():
                mask_overlay[segments == segment_id] = [1, 0, 0]  # Red color

        # Apply the mask overlay
        labeled_img = img.copy()
        labeled_img[mask_overlay.sum(axis=-1) > 0] = labeled_img[mask_overlay.sum(axis=-1) > 0] * 0.5 + mask_overlay[mask_overlay.sum(axis=-1) > 0] * 0.5

        # Calculate the total leaf area
        gray_img = color.rgb2gray(img)
        leaf_mask = gray_img > 0.3 # Threshold to identify the leaf
        total_leaf_area = np.sum(leaf_mask)

        # Calculate diseased area (number of red pixels in the mask_overlay)
        diseased_area = np.sum(mask_overlay[:, :, 0] == 1)

        # Calculate the percentage of leaf area diseased
        percentage_diseased = (diseased_area / total_leaf_area) * 100

        # Visualization
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        plt.title('Original')
        plt.imshow(img)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title('Segmentation')
        plt.imshow(mark_boundaries(img, segments))
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title(f'Diseased Segments - {percentage_diseased:.2f}% of leaf area')
        plt.imshow(labeled_img)
        plt.axis('off')

        plt.show()

# Usage
img_list = ['Browon Spot_0001']
# Start timing
start_time = time.process_time()
get_explanations(img_list, num_samples=DEFAULT_NUM_SAMPLES, random_state=0)
end_time = time.process_time()
print(f"CPU time for get_explanations: {end_time - start_time} seconds")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
from skimage.segmentation import watershed
from skimage.filters import sobel
from skimage import color
from skimage.measure import label, regionprops

def get_explanations(image_names, num_samples=DEFAULT_NUM_SAMPLES, random_state=0):
    for image_name in image_names:
        explainer = lime_image.LimeImageExplainer(random_state=random_state)
        class_name = image_name.rsplit('_', 1)[0]
        file_path = f'/kaggle/input/bd-rice-leaf-dataset/Noised Field Background/{class_name}/{image_name}.jpg'

        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = cv2.resize(img, (IMAGE_SIZE[0], IMAGE_SIZE[1]))
        img = img / 255.0

        prob = model.predict(tf.expand_dims(img, axis=0))
        prd = np.argmax(prob, axis=-1)

        # Watershed segmentation
        gradient = sobel(rgb2gray(img))
        segments = watershed(gradient, markers=250, compactness=0.001)

        explanation = explainer.explain_instance(img, model.predict, num_samples=num_samples, segmentation_fn=lambda x: segments)
        temp, mask = explanation.get_image_and_mask(prd[0], positive_only=True, num_features=5, hide_rest=False, min_weight=0.01)

        # Overlay explanation mask on the original image
        highlighted_img = img.copy()
        highlighted_img[mask == 1] = [1, 0, 0]  # Highlight the influenced regions in red

        # Visualization
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        plt.title('Original')
        plt.imshow(img)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title('Segmentation')
        plt.imshow(mark_boundaries(img, segments))
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title('Influenced Regions')
        plt.imshow(highlighted_img)
        plt.axis('off')

        plt.show()

# Usage
img_list = ['Browon Spot_0001']
# Start timing
start_time = time.process_time()
get_explanations(img_list, num_samples=DEFAULT_NUM_SAMPLES, random_state=0)
end_time = time.process_time()
print(f"CPU time for get_explanations: {end_time - start_time} seconds")