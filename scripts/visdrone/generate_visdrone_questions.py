import os
import random
import pandas as pd
from PIL import Image

IMAGE_DIR = "/home/brisic03/VisDrone/VisDrone2019-DET-val/images"
ANN_DIR = "/home/brisic03/VisDrone/VisDrone2019-DET-val/annotations"
OUT_PATH = "/home/brisic03/visdrone_val_questions.csv"

MIN_AREA_RATIO = 0.001
MAX_IMAGES = None

CLASS_NAMES = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}

LETTERS = ["A", "B", "C", "D", "E"]
random.seed(42)


def read_annotations(ann_path, image_area):
    objects = []

    with open(ann_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue

            x, y, w, h = map(float, parts[:4])
            category = int(parts[5])

            if category not in CLASS_NAMES:
                continue

            bbox_area = w * h
            area_ratio = bbox_area / image_area

            if area_ratio < MIN_AREA_RATIO:
                continue

            objects.append({
                "class_id": category,
                "class_name": CLASS_NAMES[category],
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "area_ratio": area_ratio,
            })

    return objects


def make_options(correct, candidates):
    options = [correct]
    remaining = [c for c in candidates if c != correct]
    random.shuffle(remaining)

    options.extend(remaining[:4])

    if len(options) < 5:
        return None, None

    random.shuffle(options)
    answer_idx = options.index(correct)

    return options, answer_idx


def add_question(rows, image_name, image_path, qtype, question, options, answer_idx, answer_text):
    row = {
        "image": image_name,
        "image_path": image_path,
        "qtype": qtype,
        "question": question,
        "answer": answer_idx,
        "answer_letter": LETTERS[answer_idx],
        "answer_text": answer_text,
    }

    for i, opt in enumerate(options):
        row[f"a{i}"] = opt

    rows.append(row)


image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

if MAX_IMAGES is not None:
    image_files = image_files[:MAX_IMAGES]

rows = []
all_classes = list(CLASS_NAMES.values())

for image_name in image_files:
    image_path = os.path.join(IMAGE_DIR, image_name)
    ann_name = os.path.splitext(image_name)[0] + ".txt"
    ann_path = os.path.join(ANN_DIR, ann_name)

    if not os.path.exists(ann_path):
        continue

    img = Image.open(image_path)
    img_w, img_h = img.size
    image_area = img_w * img_h

    objects = read_annotations(ann_path, image_area)

    if not objects:
        continue

    counts = {}
    for obj in objects:
        counts[obj["class_name"]] = counts.get(obj["class_name"], 0) + 1

    present_classes = list(counts.keys())

    # Question 1: object presence
    correct_class = random.choice(present_classes)
    options, answer_idx = make_options(correct_class, all_classes)

    if options is not None:
        add_question(
            rows,
            image_name,
            image_path,
            "presence",
            "Which object type is visible in the drone image?",
            options,
            answer_idx,
            correct_class,
        )

    # Question 2: counting with buckets
    target_class = random.choice(present_classes)
    count = counts[target_class]

    count_options = ["0", "1", "2", "3", "4 or more"]
    if count >= 4:
        correct_count = "4 or more"
    else:
        correct_count = str(count)

    answer_idx = count_options.index(correct_count)

    add_question(
        rows,
        image_name,
        image_path,
        "counting",
        f"How many {target_class}s are visible in the drone image?",
        count_options,
        answer_idx,
        correct_count,
    )

    # Question 3: most frequent object class
    max_count = max(counts.values())
    top_classes = [cls for cls, c in counts.items() if c == max_count]

    if len(top_classes) == 1:
        correct_top = top_classes[0]
        options, answer_idx = make_options(correct_top, all_classes)

        if options is not None:
            add_question(
                rows,
                image_name,
                image_path,
                "most_frequent",
                "Which object type appears most often in the drone image?",
                options,
                answer_idx,
                correct_top,
            )

    # Question 4: location of largest object
    largest = max(objects, key=lambda o: o["w"] * o["h"])
    cx = largest["x"] + largest["w"] / 2
    cy = largest["y"] + largest["h"] / 2

    if cx < img_w / 3 and cy < img_h / 3:
        loc = "top-left"
    elif cx > 2 * img_w / 3 and cy < img_h / 3:
        loc = "top-right"
    elif cx < img_w / 3 and cy > 2 * img_h / 3:
        loc = "bottom-left"
    elif cx > 2 * img_w / 3 and cy > 2 * img_h / 3:
        loc = "bottom-right"
    else:
        loc = "center"

    loc_options = ["top-left", "top-right", "bottom-left", "bottom-right", "center"]
    answer_idx = loc_options.index(loc)

    add_question(
        rows,
        image_name,
        image_path,
        "location",
        f"Where is the largest {largest['class_name']} located in the image?",
        loc_options,
        answer_idx,
        loc,
    )

df = pd.DataFrame(rows)
df.to_csv(OUT_PATH, index=False)

print(f"Saved {len(df)} questions to {OUT_PATH}")
print(df["qtype"].value_counts())
print(df.head())

