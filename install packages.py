#!/usr/bin/env python
import subprocess
import sys
import time
import os

def get_python_executable():
    """
    Если программа запущена как скомпилированный exe, то sys.executable будет указывать на этот exe.
    Попробуем вызвать 'python' (который должен быть в PATH) для установки пакетов.
    Если это не удаётся, то возвращаем sys.executable.
    """
    exe = sys.executable
    basename = os.path.basename(exe).lower()
    if basename != "python.exe":
        try:
            output = subprocess.check_output(["python", "--version"], stderr=subprocess.STDOUT, universal_newlines=True)
            # Если вызов успешный, используем 'python'
            return "python"
        except Exception:
            # Если не удалось, возвращаем sys.executable
            return exe
    return exe

python_exe = get_python_executable()

packages = [
    "certifi==2025.1.31",
    "charset-normalizer==3.4.1",
    "contourpy==1.3.1",
    "cycler==0.12.1",
    "filelock==3.17.0",
    "fonttools==4.55.8",
    "fsspec==2025.2.0",
    "gitdb==4.0.12",
    "GitPython==3.1.44",
    "idna==3.10",
    "Jinja2==3.1.5",
    "kiwisolver==1.4.8",
    "MarkupSafe==3.0.2",
    "matplotlib==3.10.0",
    "mpmath==1.3.0",
    "networkx==3.4.2",
    "numpy==2.1.1",
    "nvidia-cublas-cu12==12.4.5.8",
    "nvidia-cuda-cupti-cu12==12.4.127",
    "nvidia-cuda-nvrtc-cu12==12.4.127",
    "nvidia-cuda-runtime-cu12==12.4.127",
    "nvidia-cudnn-cu12==9.1.0.70",
    "nvidia-cufft-cu12==11.2.1.3",
    "nvidia-curand-cu12==10.3.5.147",
    "nvidia-cusolver-cu12==11.6.1.9",
    "nvidia-cusparse-cu12==12.3.1.170",
    "nvidia-cusparselt-cu12==0.6.2",
    "nvidia-nccl-cu12==2.21.5",
    "nvidia-nvjitlink-cu12==12.4.127",
    "nvidia-nvtx-cu12==12.4.127",
    "opencv-python==4.11.0.86",
    "packaging==24.2",
    "pandas==2.2.3",
    "pillow==11.1.0",
    "psutil==6.1.1",
    "py-cpuinfo==9.0.0",
    "pyparsing==3.2.1",
    "PyQt6==6.8.0",
    "PyQt6-Qt6==6.8.1",
    "PyQt6_sip==13.10.0",
    "python-dateutil==2.9.0.post0",
    "pytz==2025.1",
    "PyYAML==6.0.2",
    "requests==2.32.3",
    "scipy==1.15.1",
    "seaborn==0.13.2",
    "six==1.17.0",
    "smmap==5.0.2",
    "sympy==1.13.1",
    "torch==2.6.0",
    "torchvision==0.21.0",
    "tqdm==4.67.1",
    "triton==3.2.0",
    "typing_extensions==4.12.2",
    "tzdata==2025.1",
    "ultralytics==8.3.70",
    "ultralytics-thop==2.0.14",
    "urllib3==2.3.0"
]

total = len(packages)
print("\nНачинается установка {} пакетов...\n".format(total))

success = []
failed = []

try:
    for i, pkg in enumerate(packages, start=1):
        print("[{}/{}] Устанавливается: {}".format(i, total, pkg))
        start_time = time.time()
        try:
            subprocess.check_call(
                [python_exe, "-m", "pip", "install", "--quiet", pkg],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            elapsed = time.time() - start_time
            print("✓ Установлено: {} за {:.1f} сек.\n".format(pkg, elapsed))
            success.append(pkg)
        except subprocess.CalledProcessError:
            print("✗ Ошибка при установке: {}\n".format(pkg))
            failed.append(pkg)
except KeyboardInterrupt:
    print("\nУстановка прервана пользователем.\n")
except Exception as e:
    print("Произошла ошибка: {}\n".format(e))

print("Установка завершена.")
print("Успешно установлено: {} пакетов".format(len(success)))
if failed:
    print("Ошибки при установке: {} пакетов:".format(len(failed)))
    for pkg in failed:
        print("  - {}".format(pkg))
else:
    print("Все пакеты установлены успешно!")

input("\nНажмите Enter для выхода...")
