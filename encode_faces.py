import face_recognition
import os
import pickle

# Folder with known images
KNOWN_IMAGES_DIR = "images"
ENCODINGS_FILE = "encodings.pkl"

import sys

known_encodings = []
known_names = []

# Load existing encodings if file exists
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as f:
        known_encodings, known_names = pickle.load(f)

# Determine if we are training a specific person
target_person = sys.argv[1] if len(sys.argv) > 1 else None

if target_person:
    print(f"[*] Training only for: {target_person}")
    # Remove existing encodings for this person to prevent duplicates
    indices_to_keep = [i for i, name in enumerate(known_names) if name != target_person]
    known_encodings = [known_encodings[i] for i in indices_to_keep]
    known_names = [known_names[i] for i in indices_to_keep]
else:
    print("[*] Training all faces from scratch...")
    known_encodings = []
    known_names = []

for item in os.listdir(KNOWN_IMAGES_DIR):
    item_path = os.path.join(KNOWN_IMAGES_DIR, item)
    
    # If a target person is specified, skip other folders/files
    if target_person and item != target_person and os.path.splitext(item)[0] != target_person:
        continue
    
    # If it's a directory, use the directory name as the person's name
    if os.path.isdir(item_path):
        name = item
        for filename in os.listdir(item_path):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(item_path, filename)
                try:
                    image = face_recognition.load_image_file(path)
                    encodings = face_recognition.face_encodings(image)
            
                    if encodings:
                        known_encodings.append(encodings[0])
                        known_names.append(name)
                        print(f"[+] Encoded: {name} (from {filename})")
                    else:
                        print(f"[!] No face found in {filename}")
                except Exception as e:
                    print(f"[!] Error processing {filename}: {e}")

    # If it's a file, use the filename without extension as the person's name
    elif os.path.isfile(item_path) and item.lower().endswith((".jpg", ".png", ".jpeg")):
        name = os.path.splitext(item)[0]
        path = item_path
        try:
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
    
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(name)
                print(f"[+] Encoded: {name}")
            else:
                print(f"[!] No face found in {item}")
        except Exception as e:
            print(f"[!] Error processing {item}: {e}")

# Save encodings
with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump((known_encodings, known_names), f)
print(f"[+] Encodings saved. Total known faces: {len(known_encodings)}")
