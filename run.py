import subprocess
import sys

def main():
    processes = []

    # Запуск uvicorn
    p1 = subprocess.Popen([
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--reload"
    ])
    processes.append(p1)

    # Запуск воркера
    p2 = subprocess.Popen([
        sys.executable, "-m", "app.worker"
    ])
    processes.append(p2)

    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        for p in processes:
            p.terminate()

if __name__ == "__main__":
    main()