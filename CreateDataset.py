import os
import pickle
import mediapipe as mp
import cv2
import warnings
import numpy as np
from sklearn.utils import shuffle

warnings.filterwarnings("ignore", category=UserWarning)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Language mapping with number of classes (updated from collecting.py)
LANGUAGES = {
    'ISL': 28,  # Indian Sign Language
    'ASL': 28,  # American Sign Language
    'BANZSL': 28,  # British, Australian, and New Zealand Sign Language
    'DGS': 32,  # German Sign Language
    'LSF': 28,  # French Sign Language
    'ArSL': 32  # Arabic Sign Language
}

DATA_DIR = './data'

# Prompt user to select a language dataset
print("Choose the language dataset to create:")
for idx, lang in enumerate(LANGUAGES.keys(), 1):
    print(f"{idx}. {lang}")

language_choice = int(input("Enter the choice (1/2/3/4/5/6): "))
selected_language = list(LANGUAGES.keys())[language_choice - 1]

# Define path and initialize data/labels
lang_dir = os.path.join(DATA_DIR, selected_language)
data = []
labels = []

# Define maximum length for features (84 for two hands)
MAX_FEATURES = 84

# Map class labels to indices
label_to_index = {label: idx for idx, label in enumerate(sorted(os.listdir(lang_dir), key=lambda x: int(x)))}

# Process images for each class
for dir_ in os.listdir(lang_dir):
    class_index = label_to_index[dir_]
    print(f"Processing {selected_language} - class {dir_} -> {class_index}")
    for img_path in os.listdir(os.path.join(lang_dir, dir_)):
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(lang_dir, dir_, img_path))
        if img is None:
            print(f"Warning: Unable to read {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append((x - min(x_)) / (max(x_) - min(x_)))
                    data_aux.append((y - min(y_)) / (max(y_) - min(y_)))

            if len(data_aux) < MAX_FEATURES:
                data_aux.extend([0] * (MAX_FEATURES - len(data_aux)))
            elif len(data_aux) > MAX_FEATURES:
                data_aux = data_aux[:MAX_FEATURES]

            data.append(data_aux)
            labels.append(class_index)
        else:
            print(f"Warning: No hands detected in {img_path}")

# Shuffle and save dataset
data, labels = shuffle(data, labels, random_state=42)
output_file = f'{selected_language}_data.pickle'
with open(output_file, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Dataset saved for {selected_language} in {output_file}. Total samples: {len(data)}")
