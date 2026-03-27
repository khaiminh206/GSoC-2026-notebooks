# ML4SCI DeepLense - GSoC Evaluation Tests 

This repository contains my solutions for the **DeepLense** evaluation tests for Google Summer of Code (GSoC). The project focuses on identifying and classifying gravitational lenses and dark matter substructures using Deep Learning techniques.

##  Repository Structure
* `Common_Test_I.ipynb`: Solution for the multi-class classification of dark matter substructures.
* `Specific_Test_V.ipynb`: Solution for identifying lenses in a highly imbalanced dataset.
* `images/`: Directory containing evaluation plots (ROC curves).

---

##  1. Common Test I: Dark Matter Substructure Classification

**Task:** Classify simulated gravitational lensing images into three categories: `no substructure`, `spherical subhalo`, and `vortex substructure`.

### Methodology
* **Architecture:** Transfer Learning using a pre-trained **ResNet-18**.
* **Input Modification:** Adapted the first convolutional layer (`conv1`) to accept 1-channel (grayscale) physical `.npy` data instead of 3-channel RGB images.
* **Optimization:** Adam optimizer with Weight Decay to prevent overfitting.

###  Results
The model successfully distinguished between the three subtle physical structures with high confidence.
* **Evaluation Metric:** One-vs-Rest ROC Curve and AUC Score.

* **Result after 10 batches:** 
![ROC test 1](/Users/macos/Desktop/GSoC-2026-notebooks/roc_test.png)
##  2. Specific Test V: Lens Finding & Data Pipelines

**Task:** Build a model to identify lenses (Lenses vs. Non-lenses) from observational data across three different filters (shape: 3, 64, 64). 

###  Methodology & Strategy
The primary challenge of this test is the **highly imbalanced nature of the universe** (Non-lenses vastly outnumber Lenses). To tackle this, I implemented a two-fold strategy:

1. **Algorithm-Level (Weighted Loss):** Implemented a **Weighted Cross-Entropy Loss** function. By dynamically calculating the class distribution, the model applies a much heavier penalty for misclassifying the rare `Lenses` class, forcing the network to prioritize the minority class.
2. **Data-Level (Augmentation):** Utilized robust Data Augmentation (Random Horizontal/Vertical Flips) specifically during training to help the model learn spatial invariance and prevent overfitting on the limited `Lenses` samples.
3. **Architecture:** Utilized **ResNet-18** (default 3-channel input), upscaling the 64x64 images to 128x128 within the pipeline to better leverage the pre-trained spatial feature extractors.

###  Results
The combination of Weighted Loss and Transfer Learning yielded an exceptional ability to detect rare gravitational lenses without being biased towards the majority class.
* **Evaluation Metric:** Binary ROC Curve and AUC Score.
* **Result after 5 batches:** 
![ROC test 2](/Users/macos/Desktop/GSoC-2026-notebooks/roc_test2.png)

