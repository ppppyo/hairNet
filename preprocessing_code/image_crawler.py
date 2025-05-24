import os
import sys
import time
import urllib.request
import urllib.error
import shutil
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

# 실행 인자 처리
if len(sys.argv) < 4:
    print("사용법: python3 script.py <검색어> <파일이름접두어> <이미지개수>")
    sys.exit(1)

search_query = sys.argv[1]
filename_prefix = sys.argv[2]

try:
    num_images_to_download = int(sys.argv[3])
except ValueError:
    print("이미지 개수는 정수여야 합니다.")
    sys.exit(1)

# 디렉토리 생성
save_dir = f"./{filename_prefix}_imgs"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 크롬 브라우저 열기
driver = webdriver.Chrome()
driver.maximize_window()
driver.get("https://www.google.com/imghp")
driver.implicitly_wait(10)

# 검색어 입력
search_box = driver.find_element(By.CSS_SELECTOR, '#APjFqb')
search_box.send_keys(search_query)
search_box.send_keys(Keys.RETURN)
time.sleep(2)

# 이미지 수집
collected_links = []
seen_links = set()
start_index = 0
SCROLL_PAUSE_TIME = 2

while len(collected_links) < num_images_to_download:
    # 썸네일들 가져오기
    thumbnails = driver.find_elements(By.XPATH, "//div[@class='H8Rx8c']")
    print(f"▶ 썸네일 수: {len(thumbnails)} / 수집: {len(collected_links)}")

    for i in range(start_index, len(thumbnails)):
        if len(collected_links) >= num_images_to_download:
            break
        try:
            thumbnails[i].click()
            time.sleep(1)

            # 고해상도 이미지 요소 찾기
            image = driver.find_element(By.XPATH,
                '//*[@id="Sva75c"]/div[2]/div[2]/div/div[2]/c-wiz/div/div[2]/div[1]/a/img[1]')
            srcset = image.get_attribute('srcset')

            if srcset:
                urls = srcset.split(',')
                high_res_url = urls[-1].split(' ')[0]
            else:
                high_res_url = image.get_attribute('src')

            if high_res_url and high_res_url not in seen_links:
                seen_links.add(high_res_url)
                collected_links.append(high_res_url)
                print(f"✔ 수집됨: {high_res_url}")

        except Exception as e:
            print(f"✘ 이미지 로드 실패: {e}")

    start_index = len(thumbnails)

    # 스크롤 다운
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(SCROLL_PAUSE_TIME)

# 이미지 다운로드
for i, url in enumerate(collected_links):
    file_path = f"{save_dir}/{filename_prefix}_{i}.jpg"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"}  # 차단 우회용 헤더
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            with open(file_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        print(f"✔ 다운로드 성공: {file_path}")
    except urllib.error.HTTPError as e:
        print(f"✘ 다운로드 실패 (HTTPError {e.code}): {url}")
    except urllib.error.URLError as e:
        print(f"✘ 다운로드 실패 (URLError): {e.reason} - {url}")
    except Exception as e:
        print(f"✘ 알 수 없는 오류: {e} - {url}")
    time.sleep(1)  # 너무 빠른 요청 방지

print(f"\n✅ 총 {len(collected_links)}개의 이미지를 성공적으로 다운로드했습니다.")
driver.quit()

