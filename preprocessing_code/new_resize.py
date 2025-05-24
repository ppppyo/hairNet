import os
import sys
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError
from facenet_pytorch import MTCNN

# 1. 인자 확인
if len(sys.argv) < 2:
    print("사용법: python3 new_resize.py <입력_디렉토리>")
    sys.exit(1)

input_dir = sys.argv[1]
if not os.path.exists(input_dir):
    print(f"입력 디렉토리가 존재하지 않습니다: {input_dir}")
    sys.exit(1)

output_dir = input_dir.rstrip('/\\') + '_256'
os.makedirs(output_dir, exist_ok=True)

# 2. 얼굴 검출기 초기화
mtcnn = MTCNN(keep_all=True)

# 3. 흐릿한 이미지 판단 함수
def is_blurry(pil_img, threshold=100.0):
    cv_img = np.array(pil_img.convert('L'))
    variance = cv2.Laplacian(cv_img, cv2.CV_64F).var()
    return variance < threshold

# 4. 이미지 처리 함수
def process_image(file_path, save_path, target_size=(256, 256), padding_ratio=1.0):
    try:
        img = Image.open(file_path).convert('RGB')
    except UnidentifiedImageError:
        print(f"[열기 실패 - 이미지 손상 또는 포맷 오류] {file_path}")
        return
    except Exception as e:
        print(f"[예상치 못한 오류] {file_path} -> {e}")
        return

    if is_blurry(img):
        print(f"[흐릿함 - 스킵] {file_path}")
        return

    w, h = img.size
    boxes, probs = mtcnn.detect(img)

    if boxes is None or len(boxes) != 1:
        print(f"[얼굴 없음 또는 여러 명 - 스킵] {file_path}")
        return

    (x1, y1, x2, y2) = boxes[0]
    face_w = x2 - x1
    face_h = y2 - y1

    padding_w = face_w * padding_ratio
    padding_h = face_h * padding_ratio

    left = max(0, int(x1 - padding_w))
    top = max(0, int(y1 - padding_h))
    right = min(w, int(x2 + padding_w))
    bottom = min(h, int(y2 + padding_h))

    img_cropped = img.crop((left, top, right, bottom))
    img_resized = img_cropped.resize(target_size, Image.BILINEAR)
    img_resized.save(save_path)
    print(f"[얼굴 크롭 성공] {file_path} -> {save_path}")

# 5. 디렉토리 전체 순회
for filename in os.listdir(input_dir):
    if filename.lower().endswith('.jpg'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        process_image(input_path, output_path)

