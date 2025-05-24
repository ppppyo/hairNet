import os
import sys
import shutil
from ultralytics import YOLO
from PIL import UnidentifiedImageError

# 명령줄 인자 확인
if len(sys.argv) < 2:
    print("사용법: python3 filter_nonhuman_images_recursive.py <상위_디렉토리>")
    sys.exit(1)

root_dir = sys.argv[1]
if not os.path.exists(root_dir):
    print(f"[오류] 디렉토리가 존재하지 않음: {root_dir}")
    sys.exit(1)

# YOLO 모델 불러오기
model = YOLO("yolov8n.pt")  # 처음 실행 시 자동 다운로드됨

def contains_person(img_path):
    try:
        results = model(img_path)
    except UnidentifiedImageError:
        print(f"[열기 실패] {img_path}")
        return False

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            if model.model.names[cls_id] == "person":
                return True
    return False

# 하위 디렉토리 순회
for subdir, dirs, files in os.walk(root_dir):
    if subdir.endswith("to_remove"):
        continue  # 이미 to_remove 디렉토리는 건너뜀

    to_remove_dir = os.path.join(subdir, "to_remove")
    os.makedirs(to_remove_dir, exist_ok=True)

    for fname in files:
        if not fname.lower().endswith(".jpg"):
            continue

        fpath = os.path.join(subdir, fname)
        if contains_person(fpath):
            print(f"[사람 있음 유지] {fpath}")
        else:
            shutil.move(fpath, os.path.join(to_remove_dir, fname))
            print(f"[사람 없음 이동] {fpath} -> {to_remove_dir}/")

