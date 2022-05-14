import os


def get_files_recursively(data_dir, exts=[".png", ".jpg", ".jpeg"]):
    files = []
    for fpathe, dirs, fs in os.walk(data_dir):
        for f in fs:
            if os.path.splitext(f)[-1] in exts:
                files.append(os.path.join(fpathe, f))

    return files


if __name__ == "__main__":
    files = get_files_recursively("/data/docker/pengyuyan/dataset/google_image_downloader/furiends")
    print(len(files))