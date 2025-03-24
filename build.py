import os
import shutil
import subprocess


def build_exe():
    script_name = "release.py"
    output_dir = "dist"
    build_dir = "build"
    exe_name = "EzLook YOLO Detector"  # Имя выходного EXE-файла
    icon_path = "img/icon.ico"
    version_file = "version.txt"
    extra_dirs = ["img", "yolo_models"]
    requirements_file = "requirements.txt"

    # Удаляем старые сборки, если они есть
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)

    # Формируем команду для PyInstaller
    pyinstaller_cmd = [
        "pyinstaller",
        "--onefile",  # Собираем в один исполняемый файл
        "--windowed",  # Без терминала (убираем консоль)
        f"--icon={icon_path}",  # Устанавливаем иконку
        f"--version-file={version_file}",  # Указываем файл с мета-информацией
        f"--name={exe_name}",  # Указываем имя выходного EXE-файла
        "--distpath", output_dir,  # Куда сохранить EXE
        "--exclude-module", "onnx.reference",
        script_name
    ]

    print("Сборка EXE...")
    subprocess.run(pyinstaller_cmd, check=True)

    # Перемещаем дополнительные папки в dist
    for folder in extra_dirs:
        src = folder
        dst = os.path.join(output_dir, folder)
        if os.path.exists(src):
            shutil.copytree(src, dst)

    # Копируем файл requirements.txt (если он есть)
    if os.path.exists(requirements_file):
        shutil.copy(requirements_file, output_dir)

    print("Сборка завершена! Файлы находятся в папке:", output_dir)


if __name__ == "__main__":
    build_exe()
