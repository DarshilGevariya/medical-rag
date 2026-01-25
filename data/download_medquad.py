import subprocess

def download():
    subprocess.run(
        ["git", "clone", "https://github.com/abachaa/MedQuAD.git"],
        check=True
    )

if __name__ == "__main__":
    download()
