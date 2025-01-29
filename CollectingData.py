import os
import cv2

# Language mapping with number of classes
LANGUAGES = {
    'ISL': 28,  # Indian Sign Language
    'ASL': 28,  # American Sign Language
    'BANZSL': 28,  # British, Australian, and New Zealand Sign Language
    'DGS': 32,  # German Sign Language
    'LSF': 28,  # French Sign Language
    'ArSL': 32  # Arabic Sign Language
}

DATA_DIR = './data'

# Ensure the data directory exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Display language options
print("Choose the language to collect data for:")
for idx, lang in enumerate(LANGUAGES.keys(), 1):
    print(f"{idx}. {lang}")

# Select language
language_choice = int(input("Enter the choice (1/2/3/4/5/6): "))
selected_language = list(LANGUAGES.keys())[language_choice - 1]

# Get the number of classes for the selected language
number_of_classes = LANGUAGES[selected_language]

dataset_size = int(input("Enter dataset size for each class: "))

cap = cv2.VideoCapture(0)

# Create directories for the selected language
lang_dir = os.path.join(DATA_DIR, selected_language)
if not os.path.exists(lang_dir):
    os.makedirs(lang_dir)

# Iterate through classes and collect data
for j in range(number_of_classes):
    class_dir = os.path.join(lang_dir, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f"Collecting data for {selected_language} - class {j}")

    # Initial display to prepare for data collection
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f"Class {j}: Press 'Q' to Start!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    # Capture images for the current class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Display the live frame
        cv2.imshow('frame', frame)

        # Save the frame as an image
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1
        print(f"Captured image {counter} for class {j}")

        # Delay between captures
        if cv2.waitKey(25) == ord('q'):
            print("Exiting capture early.")
            break

cap.release()
cv2.destroyAllWindows()
