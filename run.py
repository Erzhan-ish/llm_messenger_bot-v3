import os
import subprocess
import sys


def main():
    from app.config import settings

    processes = []

    p_web = subprocess.Popen([
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--reload",
    ])
    processes.append(p_web)

    worker_count = settings.WORKER_COUNT

    for index in range(worker_count):
        env = os.environ.copy()
        env["WORKER_ID"] = f"worker-{index + 1}"

        p_worker = subprocess.Popen(
            [sys.executable, "-m", "app.worker"],
            env=env,
        )
        processes.append(p_worker)

    if settings.PYROGRAM_ENABLED:
        for session_name in settings.pyrogram_session_list():
            env = os.environ.copy()
            env["PYROGRAM_SESSION_NAME"] = session_name
            p_pyrogram = subprocess.Popen(
                [sys.executable, "-m", "app.pyrogram_runner"],
                env=env,
            )
            processes.append(p_pyrogram)

    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        for p in processes:
            p.terminate()

        for p in processes:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
