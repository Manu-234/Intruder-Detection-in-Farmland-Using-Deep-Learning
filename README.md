# Intruder-Detection-in-Farmland-Using-Deep-Learning
An intelligent intruder detection system for farmland that uses Faster R-CNN for object detection and FaceNet-based face recognition to identify known and unknown individuals. The system triggers alerts for intruders and animals and provides evaluation through confidence analysis and confusion matrices.

## Features
- Object detection using Faster R-CNN
- Face detection using MTCNN
- Face recognition using FaceNet (InceptionResnetV1)
- Identification of known vs unknown persons
- Animal detection (cat, dog, bird)
- Performance evaluation using confusion matrix and confidence analysis
  
## Model Architecture
- **Object Detection:** Faster R-CNN (ResNet-50 FPN)
- **Face Detection:** MTCNN
- **Face Recognition:** InceptionResnetV1 (VGGFace2 pretrained)
- **Similarity Metric:** Cosine similarity on L2-normalized embeddings

## Dataset Structure
```
dataset/
├── gallery/
│   ├── Person1/
│   ├── Person2/
└── test_images/
```

## Workflow
1. Detect objects in the image
2. If person detected → extract face
3. Generate face embedding
4. Compare with gallery embeddings
5. Trigger alert if unknown person is detected
6. Evaluate performance

## Evaluation
- Confusion Matrix
- Confidence Score Visualization

## Technologies Used
- Python
- PyTorch
- Torchvision
- FaceNet-PyTorch
- Scikit-learn
- Matplotlib

## Applications
- Smart agriculture surveillance
- Farm security systems
- Rural intrusion monitoring
- Automated wildlife detection

## Author
Manya

## License
This project is licensed under the MIT License.
