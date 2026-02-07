!pip install facenet-pytorch torch torchvision pillow scikit-learn matplotlib --quiet
!pip install facenet-pytorch --quiet
!pip install torchvision --quiet
!pip install torch --quiet
!pip uninstall -y pillow
!pip install pillow==10.3.0
!pip install --upgrade facenet-pytorch torchvision torch --quiet

import os
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
import torchvision
from PIL import Image
import numpy as np
from torchvision.transforms import functional as F
import math
import shutil
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# L2 normalize
def l2_norm(x, axis=-1, eps=1e-10):
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + eps)

# Cosine similarity
def cosine_sim(a, b):
    a = a.reshape(-1)  # flatten (1,512) -> (512,)
    b = b.reshape(b.shape[0], -1)  # flatten each centroid
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return (a @ b.T).squeeze()


# Augmentations (simple)
augment_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
])

# Face crop helper using MTCNN
mtcnn = MTCNN(image_size=160, margin=14, keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


# 4) Building gallery embeddings
GALLERY_DIR = '/content/dataset/gallery'

def build_gallery_embeddings(gallery_dir=GALLERY_DIR, augment_times=8):
    gallery = {}
    for person in os.listdir(gallery_dir):
        pdir = os.path.join(gallery_dir, person)
        if not os.path.isdir(pdir):
            continue
        embeddings = []
        for fname in os.listdir(pdir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(pdir, fname)
            img = Image.open(img_path).convert('RGB')
            # get face crop
            face = mtcnn(img)
            if face is None:
                # if mtcnn fails, try center crop and still pass through resnet
                face = transforms.CenterCrop(160)(img)
                face = transforms.Resize((160,160))(face)
                face = F.to_tensor(face)
            else:
                # face is Tensor (C,H,W) in range [0,1]
                pass
            # face -> embedding
            with torch.no_grad():
                tensor = face.unsqueeze(0).to(device)
                emb = resnet(tensor).cpu().numpy().reshape(-1)
                embeddings.append(emb)
            # augmentation
            for i in range(augment_times):
                aug = augment_transforms(img)
                face2 = mtcnn(aug)
                if face2 is None:
                    face2 = transforms.CenterCrop(160)(aug)
                    face2 = transforms.Resize((160,160))(face2)
                    face2 = F.to_tensor(face2)
                with torch.no_grad():
                    emb2 = resnet(face2.unsqueeze(0).to(device)).cpu().numpy().reshape(-1)
                    embeddings.append(emb2)
        if len(embeddings) == 0:
            print(f'No faces found for {person}, skipping')
            continue
        embeddings = np.stack(embeddings)
        embeddings = l2_norm(embeddings, axis=1)
        centroid = embeddings.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
        gallery[person] = {
            'centroid': centroid,
            'examples': embeddings
        }
        print(f'Built gallery for {person}: {embeddings.shape[0]} embeddings')
    return gallery

# 5) Object detector init (Faster R-CNN)
obj_detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
obj_detector.eval()
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A',
    'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# 6) Detection + recognition pipeline for a single image
from PIL import ImageDraw, ImageFont

def detect_and_recognize(image_path, gallery, detection_threshold=0.6, face_match_threshold=0.5, save_alerts=True, save_dir='/content/alerts'):
    img = Image.open(image_path).convert('RGB')
    img_t = transforms.ToTensor()(img).to(device)
    with torch.no_grad():
        outputs = obj_detector([img_t])
    outputs = outputs[0]
    boxes = outputs['boxes'].cpu().numpy()
    scores = outputs['scores'].cpu().numpy()
    labels = outputs['labels'].cpu().numpy()

    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)

    os.makedirs(save_dir, exist_ok=True)
    alert_flag = False
    results = []

    for box, score, label in zip(boxes, scores, labels):
        if score < detection_threshold:
            continue
        cls_name = COCO_INSTANCE_CATEGORY_NAMES[label]
        x1, y1, x2, y2 = box.astype(int).tolist()
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        draw.text((x1, y1-10), f'{cls_name} {score:.2f}', fill='red')

        if cls_name == 'person':
            # crop person region and run MTCNN to get face
            person_crop = img.crop((x1, y1, x2, y2))
            face = mtcnn(person_crop)
            if face is None:
                # no face found
                results.append(('person', 'no_face_detected', (x1,y1,x2,y2)))
                alert_flag = True
                continue
            with torch.no_grad():
                emb = resnet(face.unsqueeze(0).to(device)).cpu().numpy().reshape(-1)
            emb = emb / (np.linalg.norm(emb) + 1e-10)
            # compute cosine with centroids
            names = list(gallery.keys())
            centroids = np.stack([gallery[n]['centroid'] for n in names]) if len(names)>0 else np.zeros((0,512))
            if centroids.shape[0] == 0:
                match_name = None
                max_sim = -1
            else:
                sims = cosine_sim(emb, centroids)
                max_idx = sims.argmax()
                max_sim = sims[max_idx]
                match_name = names[max_idx]
            if max_sim >= face_match_threshold:
                results.append(('person', match_name, (x1,y1,x2,y2, max_sim)))
            else:
                results.append(('person', 'unknown', (x1,y1,x2,y2, max_sim)))
                alert_flag = True
                if save_alerts:
                    crop = img.crop((x1,y1,x2,y2))
                    fname = os.path.join(save_dir, f'alert_unknown_{Path(image_path).stem}_{len(os.listdir(save_dir))+1}.jpg')
                    crop.save(fname)
        elif cls_name in ['cat', 'dog', 'bird']:
            results.append((cls_name, 'detected', (x1,y1,x2,y2, score)))
            alert_flag = True
            if save_alerts:
                crop = img.crop((x1,y1,x2,y2))
                fname = os.path.join(save_dir, f'alert_{cls_name}_{Path(image_path).stem}_{len(os.listdir(save_dir))+1}.jpg')
                crop.save(fname)
        else:
            # ignore other classes
            pass

    return img_draw, results, alert_flag

    from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import os

# Face detection and embedding extraction setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def load_gallery_embeddings(gallery_dir):
    gallery = {}
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    for person_name in os.listdir(gallery_dir):
        person_path = os.path.join(gallery_dir, person_name)
        if os.path.isdir(person_path):
            embeddings = []
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                img = Image.open(img_path).convert('RGB')
                face, _ = mtcnn(img, return_prob=True)
                if face is not None:
                    with torch.no_grad():
                        embedding = resnet(face.unsqueeze(0).to(device))
                        embeddings.append(embedding)
            if embeddings:
                mean_emb = torch.stack(embeddings).mean(0).cpu().numpy()
                mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-10)
                gallery[person_name] = {
                  'centroid': mean_emb,
                  'examples': [e.cpu().numpy() for e in embeddings]
                }
    return gallery

    img_path = '/content/dataset/test_images/scene4.jpg'
gallery = load_gallery_embeddings('/content/dataset/gallery/')
result_img, results, alert = detect_and_recognize(img_path, gallery)
display(result_img)
print(results)
print("Alert triggered:", alert)

# corrected visualization + evaluation block
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# helper to extract label and confidence robustly from result items
def extract_label_conf(item):
    """
    Accepts a single result item which may be:
      - dict: {'label': ..., 'confidence': ...}
      - tuple/list: ('person', 'Manya', (x1,y1,x2,y2, score))
      - tuple/list variations, nested
      - plain string or number
    Returns (label:str, confidence:float)
    """
    label = 'unknown'
    conf = 0.0

    # flatten nested structures up to depth 3
    def flatten(x):
        out = []
        if isinstance(x, (list, tuple)):
            for e in x:
                out.extend(flatten(e))
        else:
            out.append(x)
        return out

    # dict case
    if isinstance(item, dict):
        # common keys
        if 'label' in item:
            label = str(item.get('label', 'unknown')).lower()
        elif 'name' in item:
            label = str(item.get('name', 'unknown')).lower()
        # confidence keys
        if 'confidence' in item:
            try:
                conf = float(item['confidence'])
            except Exception:
                conf = 0.0
        elif 'score' in item:
            try:
                conf = float(item['score'])
            except Exception:
                conf = 0.0
        return label, conf

    # otherwise flatten
    flat = flatten(item)

    # collect strings and numbers
    strings = [str(x) for x in flat if isinstance(x, str)]
    numbers = [x for x in flat if isinstance(x, (int, float, np.floating, np.integer))]

    # Decide on label:
    # - Prefer the last string (often the match name or 'unknown')
    # - If no string found but there is a class name in first position, fallback to that
    if strings:
        # avoid generic tags like 'detected' if possible by picking last
        label = strings[-1].lower()
    else:
        # fallback to first element if it's not numeric
        if len(flat) > 0 and not isinstance(flat[0], (int, float, np.floating, np.integer)):
            label = str(flat[0]).lower()

    # Decide on confidence:
    if numbers:
        # take the first numeric value that looks like a probability (0<=x<=1) if available
        prob_candidates = [float(n) for n in numbers]
        prob_like = [p for p in prob_candidates if 0.0 <= p <= 1.0]
        if prob_like:
            conf = float(prob_like[0])
        else:
            # otherwise pick the first numeric converted to float
            conf = float(prob_candidates[0])
    else:
        conf = 0.0

    # cleanup label strings like 'person' followed by actual name -> try to prefer names
    # if label is generic class and there are multiple strings, try earlier strings
    generic_classes = {'person', 'cat', 'dog', 'bird', 'animal', 'detected'}
    if label in generic_classes and len(strings) >= 2:
        # prefer earlier string that isn't the generic class
        for s in reversed(strings):
            if s.lower() not in generic_classes:
                label = s.lower()
                break

    return label, float(conf)


# === Initialize results tracking ===
true_labels = []      # actual class for each image
pred_labels = []      # predicted class
conf_scores = []      # model confidence for each prediction

# === Loop through your test set ===
test_folder = '/content/dataset/test_images/'

# safety: ensure folder exists
if not os.path.isdir(test_folder):
    raise FileNotFoundError(f"Test folder not found: {test_folder}")

for img_name in sorted(os.listdir(test_folder)):
    # skip hidden files
    if img_name.startswith('.'):
        continue

    img_path = os.path.join(test_folder, img_name)

    # Extract ground truth from filename (modify this line if your naming differs)
    # e.g., known_Manya_1.jpg  or person_known_1.jpg  or bird_3.jpg
    # If filename doesn't encode label, you can set true_label manually.
    true_label = img_name.split('_')[0].lower()
    true_labels.append(true_label)

    # Run detection/recognition
    result_img, results, alert = detect_and_recognize(img_path, gallery)

    # results may be empty list or list of tuples/dicts
    if results and len(results) > 0:
        # use the first detection result for evaluation (you can change this logic if needed)
        first = results[0]
        pred_label, conf = extract_label_conf(first)
    else:
        pred_label, conf = 'unknown', 0.0

    pred_labels.append(pred_label)
    conf_scores.append(conf)

    print(f"Image: {img_name} | True: {true_label} | Predicted: {pred_label} | Confidence: {conf:.2f} | Alert: {alert}")


# === Confusion Matrix ===
labels = sorted(list(set(true_labels + pred_labels)))  # all labels present
cm = confusion_matrix(true_labels, pred_labels, labels=labels)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# === Classification Report ===
#print(classification_report(true_labels, pred_labels, zero_division=0))

# === Confidence Line Plot ===
plt.figure(figsize=(8,4))
plt.plot(range(1, len(conf_scores)+1), conf_scores, marker='o', color='green', linewidth=2)
plt.ylim(0, 1.05)
plt.xlabel('Test Image Index')
plt.ylabel('Confidence Score')
plt.title('Model Confidence per Detection')
plt.grid(True)
plt.show()

