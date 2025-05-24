import os

def download_data():
    repo_url = "https://github.com/ppppyo/hairNet.git"
    target_dir = "data/hairNet"

    if not os.path.exists(target_dir):
        os.makedirs("data", exist_ok=True)
        os.system(f"git clone {repo_url} {target_dir}")
    else:
        print("이미 다운로드된 데이터입니다.")

if __name__ == "__main__":
    download_data()