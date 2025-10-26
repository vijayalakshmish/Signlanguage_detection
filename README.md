🧠 Real-Time Sign Language Detection using DETR

A real-time Sign Language Detection System built using the Detection Transformer (DETR) architecture.
This project demonstrates how Transformers can be applied to computer vision tasks like gesture recognition — combining CNN-based feature extraction and attention-based object detection for accessibility-driven AI.

🚀 Overview

The project captures real-time webcam input, detects hand gestures representing sign language symbols, and displays bounding boxes with classification labels using a fine-tuned DETR model.

⚙️ Tech Stack

Framework: PyTorch

Architecture: DETR (ResNet-50 + Transformer Encoder–Decoder)

Image Augmentation: Albumentations

Visualization & Camera: OpenCV

Utilities: NumPy, Matplotlib, Colorama

Loss Matching: Hungarian Algorithm (Scipy)

🧩 Project Structure
├── data.py          # Dataset loader and augmentation
├── model.py         # DETR architecture (ResNet + Transformer)
├── loss.py          # Hungarian Matcher and DETR loss computation
├── train.py         # Training loop with optimizer, scheduler, logging
├── test.py          # Inference and evaluation
├── realtime.py      # Real-time detection using webcam
├── checkpoints/     # Trained model weights
└── utils/           # Supporting modules (logger, boxes, handlers)

📦 Installation
# Clone the repository
git clone https://github.com/<yourusername>/SignDETR.git
cd SignDETR

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

📸 Running Real-Time Detection
# Ensure webcam is connected
uv run src/realtime.py


🎥 Opens a live camera feed showing detected sign gestures with bounding boxes and confidence scores.

🧠 Model Architecture

Backbone: ResNet-50 (pretrained on ImageNet)

Transformer: Encoder–decoder attention layers

Object Queries: Learnable embeddings for each potential object

Outputs:

pred_logits → gesture class probabilities

pred_boxes → bounding box coordinates (normalized)

⚖️ Loss & Optimization

The project uses a Hungarian Matching algorithm to optimally assign predictions to targets and compute:

Cross-Entropy Loss → Gesture classification

L1 Loss → Bounding box regression

GIoU Loss → Shape/overlap correction

All combined with weighted loss balancing for stable training.

🏋️ Training
uv run src/train.py


Optimizer: Adam (lr=1e-5)

Scheduler: Cosine Annealing Warm Restarts

Epochs: 100

Logs progress with rich visualization

Auto-saves model checkpoints every 10 epochs

🧪 Testing
uv run src/test.py


Evaluates model on test dataset

Visualizes detections

Logs inference time and performance metrics

🔴 Example Output
Mode	Example
Training	Loss convergence chart
Inference	Bounding boxes with predicted signs
Real-Time	Live video feed with gesture labels
💡 Key Learnings

DETR removes the need for anchors and NMS through attention-based matching.

Learned how Hungarian matching and GIoU stabilize object detection training.

Implemented real-time inference with OpenCV integration.

Deepened understanding of Transformer architecture in computer vision.
