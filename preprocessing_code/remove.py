import os
import sys
from PIL import Image

if len(sys.argv) < 2:
    print("사용법: python3 script.py <이미지_폴더_경로>")
    sys.exit(1)

input_dir = sys.argv[1]

if not os.path.exists(input_dir):
    print(f"입력 폴더가 존재하지 않습니다: {input_dir}")
    sys.exit(1)

ratio_threshold = 1.5  # 가로/세로 비율 임계값

removed_files = []

for filename in os.listdir(input_dir):
    if filename.lower().endswith('.jpg'):
        path = os.path.join(input_dir, filename)
        try:
            with Image.open(path) as img:
                w, h = img.size
            if h == 0:  # 세로가 0인 이미지 방지
                continue
            ratio = w / h
            if ratio >= ratio_threshold:
                os.remove(path)
                removed_files.append(filename)
                print(f"삭제됨: {filename} (가로/세로 비율: {ratio:.2f})")
        except Exception as e:
            print(f"이미지 열기 실패: {filename} - {e}")

print(f"총 {len(removed_files)}개의 가로가 {ratio_threshold}배 이상 긴 이미지 삭제 완료")

