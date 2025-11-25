import os
import csv
import random
from PIL import Image, ImageDraw, ImageFont

IMAGE_WIDTH, IMAGE_HEIGHT = 512, 64  # <-- Update here to match your model and data_utils.py
DATASET_DIR = os.path.join(os.path.dirname(__file__), 'ocr_dataset', 'Synthetic')
os.makedirs(DATASET_DIR, exist_ok=True)
FONT_PATH = os.path.join(os.path.dirname(__file__), 'BrittanySignature.ttf')  # <-- Download and place a handwriting font here

# Example sentences for doctor's notes
SENTENCES = [
    "Take 2 tablets at 8:00 AM and 1 at 8:00 PM.",
    "Patient should rest and drink fluids.",
    "Next appointment: 12/11/2025 at 10:30 AM.",
    "Apply ointment twice daily.",
    "Blood pressure: 120/80 mmHg.",
    "Prescribed: Acetaminophen 500mg.",
    "Follow up in 2 weeks.",
    "No known allergies.",
    "Patient reports mild headache.",
    "Increase water intake.",
    "Administer insulin before breakfast.",
    "Patient is allergic to penicillin.",
    "Monitor temperature every 4 hours.",
    "Schedule MRI for next month.",
    "Take medication with food.",
    "Patient complains of chest pain.",
    "Refer to cardiologist for evaluation.",
    "Continue current treatment plan.",
    "Reduce dosage if side effects occur.",
    "Patient has history of hypertension.",
    "Apply ice pack to affected area.",
    "Patient should avoid strenuous activity.",
    "Check blood sugar levels daily.",
    "Patient reports dizziness and nausea.",
    "Prescribed: Ibuprofen 200mg as needed.",
    "Patient is recovering well post-surgery.",
    "No signs of infection observed.",
    "Patient advised to quit smoking.",
    "Increase dosage to 10mg daily.",
    "Patient scheduled for follow-up on 01/12/2026.",
    "Patient should wear compression stockings.",
    "Take antibiotics for 7 days.",
    "Patient has mild fever.",
    "Patient advised to reduce salt intake.",
    "Patient should return if symptoms worsen.",
    "Patient is on a low-fat diet.",
    "Patient reports improved appetite.",
    "Patient should avoid caffeine.",
    "Patient has elevated cholesterol.",
    "Patient advised to exercise regularly.",
    "Patient should take medication at bedtime.",
    "Patient has normal heart rate.",
    "Patient is experiencing joint pain.",
    "Patient should use inhaler as directed.",
    "Patient is scheduled for blood test.",
    "Patient advised to increase fiber intake.",
    "Patient should avoid dairy products.",
    "Patient has mild swelling in ankles.",
    "Patient should rest for 48 hours.",
    "Patient is prescribed vitamin D supplements.",
    "Patient should monitor weight weekly.",
    "Patient advised to keep wound clean and dry.",
    "Patient should avoid direct sunlight.",
    "Patient is experiencing mild anxiety.",
    "Patient should take pain medication as needed.",
    "Patient advised to limit sugar intake.",
    "Patient is scheduled for X-ray.",
    "Patient should drink at least 2 liters of water daily.",
    "Patient has no significant medical history.",
    "Patient should avoid alcohol.",
    "Patient is prescribed antihistamines.",
    "Patient should elevate leg when resting.",
    "Patient advised to use sunscreen.",
    "Patient should report any side effects.",
    "Patient is scheduled for physical therapy.",
    "Patient should avoid heavy lifting.",
    "Patient advised to take deep breaths.",
    "Patient should use nasal spray twice daily.",
    "Patient is prescribed antibiotics.",
    "Patient should avoid spicy foods.",
    "Patient is experiencing mild fatigue.",
    "Patient should take medication with plenty of water.",
    "Patient advised to keep hydrated.",
    "Patient should avoid crowded places.",
    "Patient is scheduled for ultrasound.",
    "Patient should take temperature daily.",
    "Patient advised to rest and recover.",
    "Patient should avoid processed foods.",
    "Patient is prescribed calcium supplements.",
    "Patient should follow up in one month.",
    "Patient advised to maintain a balanced diet.",
    "Patient should avoid unnecessary stress.",
    "Patient is experiencing mild cough.",
    "Patient should take prescribed medication regularly.",
    "Patient advised to avoid allergens.",
    "Patient should keep a symptom diary.",
    "Patient is scheduled for CT scan.",
    "Patient should avoid contact sports.",
    "Patient advised to use moisturizer.",
    "Patient should take medication before meals.",
    "Patient is prescribed iron supplements.",
    "Patient should avoid cold drinks.",
    "Patient advised to get adequate sleep.",
    "Patient should avoid fried foods.",
    "Patient is experiencing mild back pain.",
    "Patient should take medication every 8 hours.",
    "Patient advised to avoid loud noises.",
    "Patient should use prescribed eye drops.",
    "Patient is scheduled for dental checkup.",
    "Patient should avoid swimming pools.",
    "Patient advised to practice relaxation techniques.",
    "Patient should take medication for 5 days.",
    "Patient is prescribed antihypertensive drugs.",
    "Patient should avoid high-sugar foods.",
    "Patient advised to wear protective clothing.",
    "Patient should take medication after meals.",
]

MAX_LABEL_LEN = 128  # Should match feature_width in your model

def random_note():
    note = ""
    tries = 0
    while tries < 10:
        sentence = random.choice(SENTENCES)
        # Only add if it won't exceed max label length
        if len(note) + len(sentence) + 1 > MAX_LABEL_LEN:
            break
        note = (note + " " + sentence).strip() if note else sentence
        tries += 1
    return note

def create_image(text, idx):
    img = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), color=255)
    try:
        font = ImageFont.truetype(FONT_PATH, size=28)
    except Exception:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(img)
    # Center text vertically, left align
    draw.text((5, 10), text, font=font, fill=0)
    img_path = os.path.join(DATASET_DIR, f"note_{idx:05d}.png")
    img.save(img_path)
    return img_path

def clear_synthetic_folder():
    for fname in os.listdir(DATASET_DIR):
        fpath = os.path.join(DATASET_DIR, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)

def main():
    clear_synthetic_folder()
    csv_path = os.path.join(DATASET_DIR, "synthetic_labels.csv")
    with open(csv_path, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["IMAGE", "NOTE_TEXT"])
        for idx in range(10000):
            text = random_note()
            img_path = create_image(text, idx)
            writer.writerow([os.path.basename(img_path), text])
    print(f"Generated 10,000 synthetic doctor's notes and CSV at {DATASET_DIR}")

if __name__ == "__main__":
    main()