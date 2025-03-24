#!/usr/bin/env python
import os
import sys
import json
import zipfile
import requests
from pathlib import Path
import tempfile
import torch
import threading
import shutil
import time

from ultralytics import YOLO

# Попытка импортировать PyQt6 для графического интерфейса
try:
    from PyQt6 import QtWidgets, QtCore
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

SETTINGS_FILE = "settings.json"

#############################################
# Функция для автоматической конвертации аннотаций VisDrone в формат YOLO
#############################################
def visdrone2yolo(directory: Path):
    from PIL import Image
    from tqdm import tqdm

    def convert_box(size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]
        return ((box[0] + box[2] / 2) * dw,
                (box[1] + box[3] / 2) * dh,
                box[2] * dw,
                box[3] * dh)

    labels_dir = directory / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    ann_dir = directory / "annotations"
    annotation_files = list(ann_dir.glob("*.txt"))
    if not annotation_files:
        print(f"Нет аннотаций в {ann_dir}")
        return
    for f in tqdm(annotation_files, desc=f'Converting {directory.name}'):
        img_path = directory / "images" / (f.stem + ".jpg")
        try:
            img_size = Image.open(img_path).size
        except Exception as e:
            print(f"Ошибка открытия изображения {img_path}: {e}")
            continue
        lines = []
        with open(f, "r", encoding="utf-8") as file:
            for line in file.read().strip().splitlines():
                parts = line.split(',')
                if len(parts) < 6:
                    continue
                if parts[4] == '0':
                    continue
                try:
                    cls = int(parts[5]) - 1
                    box = convert_box(img_size, tuple(map(int, parts[:4])))
                except Exception as e:
                    print(f"Ошибка обработки строки в {f}: {e}")
                    continue
                lines.append(f"{cls} " + " ".join(f"{x:.6f}" for x in box) + "\n")
        label_file = labels_dir / f.name
        with open(label_file, "w", encoding="utf-8") as fl:
            fl.writelines(lines)

#############################################
# Функция для автоматической загрузки датасета VisDrone
#############################################
def download_visdrone_dataset(temp_dir):
    """
    Скачивает и распаковывает архивы датасета VisDrone из официальных URL с GitHub,
    конвертирует аннотации и генерирует YAML-конфигурацию.
    """
    os.makedirs(temp_dir, exist_ok=True)
    urls = {
        "VisDrone2019-DET-train.zip": "https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-train.zip",
        "VisDrone2019-DET-val.zip": "https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-val.zip",
        "VisDrone2019-DET-test-dev.zip": "https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-test-dev.zip",
        "VisDrone2019-DET-test-challenge.zip": "https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-test-challenge.zip",
    }
    for zip_name, url in urls.items():
        zip_path = os.path.join(temp_dir, zip_name)
        if not os.path.exists(zip_path):
            print(f"Скачивание {zip_name}...")
            try:
                r = requests.get(url, stream=True)
                r.raise_for_status()
                with open(zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"{zip_name} успешно скачан.")
            except Exception as e:
                print(f"Ошибка при скачивании {zip_name}: {e}")
                sys.exit(1)
        else:
            print(f"{zip_name} уже присутствует.")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                print(f"Распаковка {zip_name}...")
                zip_ref.extractall(temp_dir)
            print(f"Распаковка {zip_name} завершена.")
        except Exception as e:
            print(f"Ошибка при распаковке {zip_name}: {e}")
            sys.exit(1)

    for subset in ["VisDrone2019-DET-train", "VisDrone2019-DET-val", "VisDrone2019-DET-test-dev"]:
        subset_dir = Path(temp_dir) / subset
        if subset_dir.exists():
            visdrone2yolo(subset_dir)
        else:
            print(f"Папка {subset_dir} не найдена!")
    yaml_content = f"""# VisDrone2019-DET dataset configuration
# Скачано и распаковано в {temp_dir}
path: {temp_dir}
train: VisDrone2019-DET-train/images
val: VisDrone2019-DET-val/images
test: VisDrone2019-DET-test-dev/images

names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
"""
    yaml_path = os.path.join(temp_dir, "VisDrone.yaml")
    try:
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        print(f"YAML-конфигурация записана в {yaml_path}")
    except Exception as e:
        print(f"Ошибка при записи YAML-конфигурации: {e}")
        sys.exit(1)
    return yaml_path

#############################################
# Функции работы с настройками (JSON)
#############################################
def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Ошибка при загрузке настроек: {e}")
    return {}

def save_settings(settings):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4)
        print("Настройки сохранены в settings.json")
    except Exception as e:
        print(f"Ошибка при сохранении настроек: {e}")

def clear_settings():
    if os.path.exists(SETTINGS_FILE):
        os.remove(SETTINGS_FILE)
        print("Файл настроек удалён.")
    else:
        print("Файл настроек не найден.")

#############################################
# Поток для обучения (QThread) с поддержкой graceful stop
#############################################
if PYQT_AVAILABLE:
    class TrainingThread(QtCore.QThread):
        log_signal = QtCore.pyqtSignal(str)
        finished_signal = QtCore.pyqtSignal()

        def __init__(self, params):
            super().__init__()
            self.params = params  # словарь с параметрами
            self.interrupt_flag = False

        def monitor_stop(self, model):
            # Ждем, пока появится объект тренера
            while not hasattr(model, 'trainer') or model.trainer is None:
                time.sleep(0.5)
            # Пока тренировка идет, проверяем флаг остановки
            while getattr(model.trainer, "running", False):
                if self.interrupt_flag:
                    self.log_signal.emit("Принудительное завершение: установка флага остановки в тренере.")
                    model.trainer.stop = True
                    break
                time.sleep(0.5)

        def run(self):
            try:
                self.log_signal.emit("Начало обучения...")
                # Оптимизация cudnn и включение TF32 для RTX 4070
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

                mode = self.params["mode"]
                epochs = self.params["epochs"]
                imgsz = self.params["imgsz"]
                data_yaml = self.params["data_yaml"]
                project_dir = self.params["project_dir"]
                exp_name = self.params["exp_name"]
                device = self.params["device"]

                if mode == "new":
                    self.log_signal.emit("Обучение модели YOLO11x с нуля с использованием AMP...")
                    model = YOLO("yolo11x.pt")
                elif mode == "resume":
                    model_path = self.params["model_path"]
                    self.log_signal.emit("Дообучение модели с использованием AMP...")
                    model = YOLO(model_path)
                else:
                    self.log_signal.emit("Неверный режим обучения.")
                    self.finished_signal.emit()
                    return

                # Запускаем дополнительный поток для контроля прерывания
                monitor_thread = threading.Thread(target=self.monitor_stop, args=(model,))
                monitor_thread.start()

                # Запускаем обучение; обратите внимание, что теперь без параметра callbacks!
                model.train(data=data_yaml,
                            epochs=epochs,
                            imgsz=imgsz,
                            project=project_dir,
                            name=exp_name,
                            device=device,
                            workers=8,
                            amp=True)
                monitor_thread.join()
            except KeyboardInterrupt:
                self.log_signal.emit("Обучение прервано пользователем (после завершения эпохи).")
            except Exception as e:
                self.log_signal.emit(f"Ошибка обучения: {e}")

            # После завершения обучения – ищем контрольную точку
            output_dir = os.path.join(project_dir, exp_name)
            final_model_path = None
            for root, dirs, files in os.walk(output_dir):
                if "best.pt" in files:
                    final_model_path = os.path.join(root, "best.pt")
                    break
            if final_model_path is None:
                for root, dirs, files in os.walk(output_dir):
                    if "last.pt" in files:
                        final_model_path = os.path.join(root, "last.pt")
                        break
            if final_model_path is None:
                self.log_signal.emit("Не удалось найти обученную модель для экспорта.")
                self.finished_signal.emit()
                return

            # Копирование весов в целевой файл с именем, указанным пользователем
            final_name = self.params.get("final_name", exp_name)
            target_pt = os.path.join(project_dir, final_name + ".pt")
            shutil.copy(final_model_path, target_pt)
            self.log_signal.emit(f"Обученная модель сохранена в формате .pt: {target_pt}")

            self.log_signal.emit("Экспорт модели в формат ONNX...")
            try:
                # Создаем новый экземпляр модели с обученными весами
                trained_model = YOLO(target_pt)
                onnx_path = trained_model.export(format="onnx", simplify=True)
                if isinstance(onnx_path, (list, tuple)):
                    onnx_path = onnx_path[0]
                target_onnx = os.path.join(project_dir, final_name + ".onnx")
                shutil.move(onnx_path, target_onnx)
                self.log_signal.emit(f"Экспорт завершён. ONNX модель сохранена: {target_onnx}")
            except Exception as e:
                self.log_signal.emit(f"Ошибка при экспорте модели: {e}")
            self.finished_signal.emit()

#############################################
# Графический интерфейс
#############################################
if PYQT_AVAILABLE:
    class TrainingGUI(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Обучение модели Yolo | EzLook YoLo detector")
            self.resize(600, 600)
            self.settings = load_settings()
            self.train_thread = None
            self.init_ui()
            self.load_settings_into_fields()

        def init_ui(self):
            layout = QtWidgets.QVBoxLayout()
            form_layout = QtWidgets.QFormLayout()

            self.dataset_path_edit = QtWidgets.QLineEdit()
            self.browse_dataset_btn = QtWidgets.QPushButton("Обзор")
            self.browse_dataset_btn.clicked.connect(self.browse_dataset)
            dataset_layout = QtWidgets.QHBoxLayout()
            dataset_layout.addWidget(self.dataset_path_edit)
            dataset_layout.addWidget(self.browse_dataset_btn)
            form_layout.addRow("Путь к YAML датасета:", dataset_layout)

            self.model_path_edit = QtWidgets.QLineEdit()
            self.browse_model_btn = QtWidgets.QPushButton("Обзор")
            self.browse_model_btn.clicked.connect(self.browse_model)
            model_layout = QtWidgets.QHBoxLayout()
            model_layout.addWidget(self.model_path_edit)
            model_layout.addWidget(self.browse_model_btn)
            form_layout.addRow("Путь к модели (.pt) (для дообучения):", model_layout)

            self.project_dir_edit = QtWidgets.QLineEdit()
            self.browse_project_btn = QtWidgets.QPushButton("Обзор")
            self.browse_project_btn.clicked.connect(self.browse_project)
            project_layout = QtWidgets.QHBoxLayout()
            project_layout.addWidget(self.project_dir_edit)
            project_layout.addWidget(self.browse_project_btn)
            form_layout.addRow("Директория для экспериментов:", project_layout)

            self.epochs_edit = QtWidgets.QSpinBox()
            self.epochs_edit.setRange(1, 10000)
            self.epochs_edit.setValue(100)
            form_layout.addRow("Количество эпох:", self.epochs_edit)

            self.imgsz_edit = QtWidgets.QSpinBox()
            self.imgsz_edit.setRange(32, 2048)
            self.imgsz_edit.setValue(640)
            form_layout.addRow("Размер входного изображения:", self.imgsz_edit)

            self.final_name_edit = QtWidgets.QLineEdit()
            form_layout.addRow("Имя финальной модели:", self.final_name_edit)

            self.mode_new_radio = QtWidgets.QRadioButton("Новое обучение")
            self.mode_resume_radio = QtWidgets.QRadioButton("Дообучение")
            self.mode_new_radio.setChecked(True)
            mode_layout = QtWidgets.QHBoxLayout()
            mode_layout.addWidget(self.mode_new_radio)
            mode_layout.addWidget(self.mode_resume_radio)
            form_layout.addRow("Режим обучения:", mode_layout)

            layout.addLayout(form_layout)

            settings_btn_layout = QtWidgets.QHBoxLayout()
            self.load_settings_btn = QtWidgets.QPushButton("Загрузить настройки")
            self.load_settings_btn.clicked.connect(self.load_settings_into_fields)
            self.save_settings_btn = QtWidgets.QPushButton("Сохранить настройки")
            self.save_settings_btn.clicked.connect(self.save_settings_from_fields)
            self.clear_settings_btn = QtWidgets.QPushButton("Очистить настройки")
            self.clear_settings_btn.clicked.connect(self.clear_settings)
            settings_btn_layout.addWidget(self.load_settings_btn)
            settings_btn_layout.addWidget(self.save_settings_btn)
            settings_btn_layout.addWidget(self.clear_settings_btn)
            layout.addLayout(settings_btn_layout)

            btn_layout = QtWidgets.QHBoxLayout()
            self.start_btn = QtWidgets.QPushButton("Начать обучение")
            self.start_btn.clicked.connect(self.start_training)
            btn_layout.addWidget(self.start_btn)
            self.stop_btn = QtWidgets.QPushButton("Принудительно завершить обучение")
            self.stop_btn.clicked.connect(self.stop_training)
            self.stop_btn.setEnabled(False)
            btn_layout.addWidget(self.stop_btn)
            layout.addLayout(btn_layout)

            self.log_edit = QtWidgets.QTextEdit()
            self.log_edit.setReadOnly(True)
            layout.addWidget(self.log_edit)

            self.setLayout(layout)

        def browse_dataset(self):
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите YAML датасета", "", "YAML Files (*.yaml *.yml)")
            if filename:
                self.dataset_path_edit.setText(filename)

        def browse_model(self):
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите модель (.pt)", "", "PT Files (*.pt)")
            if filename:
                self.model_path_edit.setText(filename)

        def browse_project(self):
            directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите директорию для экспериментов")
            if directory:
                self.project_dir_edit.setText(directory)

        def load_settings_into_fields(self):
            self.settings = load_settings()
            if "dataset_path" in self.settings:
                self.dataset_path_edit.setText(self.settings["dataset_path"])
            if "model_path" in self.settings:
                self.model_path_edit.setText(self.settings["model_path"])
            if "project_dir" in self.settings:
                self.project_dir_edit.setText(self.settings["project_dir"])

        def save_settings_from_fields(self):
            self.settings["dataset_path"] = self.dataset_path_edit.text().strip()
            self.settings["model_path"] = self.model_path_edit.text().strip()
            self.settings["project_dir"] = self.project_dir_edit.text().strip()
            save_settings(self.settings)
            QtWidgets.QMessageBox.information(self, "Настройки", "Настройки сохранены!")

        def clear_settings(self):
            clear_settings()
            self.dataset_path_edit.clear()
            self.model_path_edit.clear()
            self.project_dir_edit.clear()
            QtWidgets.QMessageBox.information(self, "Настройки", "Настройки очищены!")

        def append_log(self, message):
            self.log_edit.append(message)

        def start_training(self):
            data_yaml = self.dataset_path_edit.text().strip()
            if not data_yaml:
                default_temp = "C:\\visdrone_dataset" if sys.platform.startswith("win") else os.path.join(tempfile.gettempdir(), "visdrone_dataset")
                default_yaml = os.path.join(default_temp, "VisDrone.yaml")
                if not os.path.exists(default_yaml):
                    self.append_log("Файл YAML датасета не найден. Скачиваем VisDrone в " + default_temp)
                    data_yaml = download_visdrone_dataset(default_temp)
                else:
                    data_yaml = default_yaml
                self.dataset_path_edit.setText(data_yaml)
            model_path = self.model_path_edit.text().strip()
            project_dir = self.project_dir_edit.text().strip()
            if not project_dir:
                project_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите директорию для экспериментов")
                self.project_dir_edit.setText(project_dir)
            epochs = self.epochs_edit.value()
            imgsz = self.imgsz_edit.value()
            mode = "new" if self.mode_new_radio.isChecked() else "resume"
            if mode == "resume" and not model_path:
                QtWidgets.QMessageBox.warning(self, "Ошибка", "Для дообучения укажите путь к модели (.pt)")
                return
            final_name = self.final_name_edit.text().strip()
            if not final_name:
                final_name = "trained_model"

            self.save_settings_from_fields()

            params = {
                "data_yaml": data_yaml,
                "model_path": model_path,
                "project_dir": project_dir,
                "epochs": epochs,
                "imgsz": imgsz,
                "mode": mode,
                "exp_name": ("new_train_exp" if mode=="new" else "resume_train_exp"),
                "device": "cuda:0",
                "final_name": final_name
            }
            self.append_log("Запуск обучения с параметрами:\n" + json.dumps(params, indent=2, ensure_ascii=False))
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)

            self.train_thread = TrainingThread(params)
            self.train_thread.log_signal.connect(self.append_log)
            self.train_thread.finished_signal.connect(self.training_finished)
            self.train_thread.start()

        def stop_training(self):
            if self.train_thread and self.train_thread.isRunning():
                self.append_log("Запрос на принудительное завершение обучения...")
                self.train_thread.interrupt_flag = True
                self.stop_btn.setEnabled(False)

        def training_finished(self):
            self.append_log("Обучение завершено.")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

    def launch_gui():
        app = QtWidgets.QApplication(sys.argv)
        window = TrainingGUI()
        window.show()
        sys.exit(app.exec())
else:
    def launch_gui():
        print("PyQt6 не установлен. Запустите программу в командной строке.")

#############################################
# Запуск программы
#############################################
if __name__ == "__main__":
    if PYQT_AVAILABLE:
        launch_gui()
    else:
        print("Графический интерфейс недоступен. Установите PyQt6 или запустите программу с --nogui.")
