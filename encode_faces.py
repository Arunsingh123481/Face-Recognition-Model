import face_recognition
import os
import pickle

# Folder with known images
KNOWN_IMAGES_DIR = "images"
ENCODINGS_FILE = "encodings.pkl"

known_encodings = []
known_names = []

for filename in os.listdir(KNOWN_IMAGES_DIR):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        path = os.path.join(KNOWN_IMAGES_DIR, filename)
        name = os.path.splitext(filename)[0]
        
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(name)
            print(f"[+] Encoded: {name}")
        else:
            print(f"[!] No face found in {filename}")

# Save encodings
with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump((known_encodings, known_names), f)
print("[✅] Encodings saved")
