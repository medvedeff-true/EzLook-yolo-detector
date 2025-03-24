import sys
import os
import cv2
import numpy as np
import torch
import time
import requests

from ultralytics import YOLO

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QRadioButton,
    QComboBox, QSlider, QFileDialog, QMessageBox, QHBoxLayout, QVBoxLayout,
    QSizePolicy, QToolButton, QPushButton, QStyle, QDialog, QCheckBox
)
from PyQt6.QtGui import QImage, QPixmap, QIcon, QMovie
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer, QSize

# ===============================
# Глобальные словари и функции
# ===============================
YOLO_VERSIONS = {
    "YOLOv3": "yolov3.pt",
    "YOLOv4": "yolov4.pt",
    "YOLOv5 Nano": "yolov5n.pt",
    "YOLOv5 Small": "yolov5s.pt",
    "YOLOv5 Medium": "yolov5m.pt",
    "YOLOv5 Large": "yolov5l.pt",
    "YOLOv8 Nano": "yolov8n.pt",  # по умолчанию
    "YOLOv8 Small": "yolov8s.pt",
    "YOLOv8 Medium": "yolov8m.pt",
    "YOLOv8 Large": "yolov8l.pt",
    "YOLOv8 XLarge": "yolov8x.pt"
}

YOLO_DESCRIPTIONS = {
    "YOLOv3": "Старая, но проверенная модель для задач детекции с невысокими требованиями к ресурсам.",
    "YOLOv4": "Улучшенная версия YOLOv3 с лучшей точностью.",
    "YOLOv5 Nano": "Очень легкая модель для быстрого, но менее точного детектирования.",
    "YOLOv5 Small": "Хороший баланс между скоростью и точностью для повседневных задач.",
    "YOLOv5 Medium": "Повышенная точность за счет большего числа параметров.",
    "YOLOv5 Large": "Высокая точность для ресурсоемких задач, требует больше вычислительных ресурсов.",
    "YOLOv8 Nano": "Экстремально легкая модель, оптимальна для мобильных устройств.",
    "YOLOv8 Small": "Стандартная версия с хорошим балансом для реального времени.",
    "YOLOv8 Medium": "Увеличенная точность за счет большего количества параметров.",
    "YOLOv8 Large": "Подходит для сложных задач, где важна высокая точность.",
    "YOLOv8 XLarge": "Максимальная точность модели, но с повышенными требованиями к ресурсам."
}

MODEL_URLS = {
    "yolov3.pt": "https://github.com/ultralytics/yolov3/releases/download/v9.5.0/yolov3.pt",
    "yolov4.pt": "https://github.com/WongKinYiu/PyTorch_YOLOv4/releases/download/0.1/yolov4.pt",
    "yolov5n.pt": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt",
    "yolov5s.pt": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt",
    "yolov5m.pt": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt",
    "yolov5l.pt": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt",
    "yolov8n.pt": "https://github.com/ultralytics/ultralytics/releases/download/v8.0/yolov8n.pt",
    "yolov8s.pt": "https://github.com/ultralytics/ultralytics/releases/download/v8.0/yolov8s.pt",
    "yolov8m.pt": "https://github.com/ultralytics/ultralytics/releases/download/v8.0/yolov8m.pt",
    "yolov8l.pt": "https://github.com/ultralytics/ultralytics/releases/download/v8.0/yolov8l.pt",
    "yolov8x.pt": "https://github.com/ultralytics/ultralytics/releases/download/v8.0/yolov8x.pt"
}


def download_file(url, dest, progress_callback):
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    downloaded = 0
    block_size = 4096
    with open(dest, 'wb') as file:
        for data in response.iter_content(block_size):
            file.write(data)
            downloaded += len(data)
            if total:
                percent = int(downloaded * 100 / total)
                progress_callback(percent)
    progress_callback(100)


# ===============================
# Классы из main.py
# ===============================

# --- ClickableSlider ---
class ClickableSlider(QSlider):
    """QSlider с возможностью клика для мгновенной установки значения."""
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            new_val = QStyle.sliderValueFromPosition(self.minimum(), self.maximum(),
                                                     int(event.position().x()), self.width())
            self.setValue(new_val)
            self.sliderReleased.emit()
        super().mousePressEvent(event)


# --- VideoThread ---
class VideoThread(QThread):
    """Поток для захвата видео, выполнения YOLO-детекции и управления воспроизведением."""
    frame_signal = pyqtSignal(np.ndarray)
    position_signal = pyqtSignal(int, int)

    def __init__(self, cap, model, is_file=False, show_fps=False, parent=None):
        super().__init__(parent)
        self.cap = cap
        self.model = model
        self.is_file = is_file
        self.show_fps = show_fps
        self._run_flag = True
        self._paused = False
        self.seek_position = None
        self.last_fps_time = time.time()
        self.frame_counter = 0
        self.current_fps = 0

    def run(self):
        if self.is_file:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            delay_sec = 1.0 / fps if fps > 0 else 1.0 / 30
        else:
            delay_sec = 1.0 / 60  # для камеры

        last_frame_time = time.time()

        while self._run_flag:
            if self._paused:
                self.msleep(10)
                last_frame_time = time.time()  # сбрасываем время, чтобы не было ускорения
                continue

            if self.seek_position is not None:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.seek_position)
                self.seek_position = None
                last_frame_time = time.time()  # сбрасываем для предотвращения резкого ускорения

            ret, frame = self.cap.read()
            if not ret:
                if self.is_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._paused = True
                    continue
                else:
                    break

            # YOLO детекция
            results = self.model(frame)
            annotated_frame = results[0].plot()
            self.frame_counter += 1
            elapsed = time.time() - self.last_fps_time
            if elapsed >= 1.0:
                self.current_fps = self.frame_counter / elapsed
                self.frame_counter = 0
                self.last_fps_time = time.time()

            if self.show_fps:
                cv2.putText(annotated_frame, f"FPS: {self.current_fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            self.frame_signal.emit(annotated_frame)

            if self.is_file:
                current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.position_signal.emit(current_frame, total_frames)

            elapsed_frame = time.time() - last_frame_time
            sleep_time = delay_sec - elapsed_frame
            if sleep_time > 0:
                self.msleep(int(sleep_time * 1000))
            last_frame_time = time.time()

        self.cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

    def toggle_pause(self):
        self._paused = not self._paused


# --- FullscreenWindow ---
class FullscreenWindow(QMainWindow):
    """Полноэкранное окно с областью видео и панелью управления."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Полный экран видео")
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setStyleSheet("background-color: black;")
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_slider = ClickableSlider(Qt.Orientation.Horizontal)
        self.video_slider.setMinimum(0)
        self.video_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.video_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 12px;
                background: #bbb;
                border-radius: 6px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                border: none;
                width: 20px;
                height: 20px;
                margin: -5px 0;
                border-radius: 10px;
            }
        """)
        self.play_pause_button = QToolButton()
        self.play_pause_button.setIcon(QIcon("img/play.png"))
        self.play_pause_button.setFixedSize(60, 60)
        self.play_pause_button.setStyleSheet("background: transparent; border: none;")
        self.minimize_button = QToolButton()
        self.minimize_button.setIcon(QIcon("img/minimize.png"))
        self.minimize_button.setFixedSize(60, 60)
        self.minimize_button.setStyleSheet("background: transparent; border: none;")
        self.minimize_button.clicked.connect(self.showMinimized)
        self.exit_button = QToolButton()
        self.exit_button.setIcon(QIcon("img/fullscreen.png"))
        self.exit_button.setFixedSize(60, 60)
        self.exit_button.setStyleSheet("background: transparent; border: none;")
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.play_pause_button)
        control_layout.addWidget(self.video_slider)
        control_layout.addWidget(self.minimize_button)
        control_layout.addWidget(self.exit_button)
        control_widget = QWidget()
        control_widget.setLayout(control_layout)
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label, stretch=1)
        main_layout.addWidget(control_widget, stretch=0)
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)


# --- CameraSelectionDialog ---
class CameraSelectionDialog(QDialog):
    """Диалог выбора веб-камеры. Сканирует индексы 0–9 и выводит список найденных устройств."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Выбрать веб-камеру")
        self.resize(300, 120)
        layout = QVBoxLayout()
        self.combo = QComboBox()
        self.cameras = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.cameras.append(i)
                self.combo.addItem(f"Камера {i}", i)
                cap.release()
        layout.addWidget(self.combo)
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Отмена")
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)

    def selected_camera(self):
        return self.combo.currentData()


# --- ModelLoaderThread ---
class ModelLoaderThread(QThread):
    """
    Поток для загрузки модели YOLO.
    Если файл модели существует локально – загружается напрямую,
    иначе – производится попытка удалённой загрузки.
    """
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)  # не используется в данном примере

    def __init__(self, model_file, device, parent=None):
        super().__init__(parent)
        self.model_file = model_file
        self.device = device

    def run(self):
        try:
            base_name = os.path.basename(self.model_file)
            if os.path.exists(self.model_file):
                model = YOLO(self.model_file)
            else:
                if base_name.startswith("yolov8"):
                    model = YOLO(self.model_file)
                elif base_name.startswith("yolov5"):
                    hub_model_name = base_name.split('.')[0]
                    model = torch.hub.load('ultralytics/yolov5', hub_model_name, pretrained=True)
                elif base_name == "yolov3.pt":
                    model = torch.hub.load('ultralytics/yolov3', 'yolov3', pretrained=True)
                elif base_name == "yolov4.pt":
                    model = torch.hub.load('WongKinYiu/PyTorch_YOLOv4', 'yolov4', pretrained=True)
                else:
                    raise Exception(f"Unsupported model: {base_name}")
            model.to(self.device)
            self.finished_signal.emit(model)
        except Exception as e:
            self.error_signal.emit(str(e))


# --- SettingsDialog ---
class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройки моделей YOLO")
        self.setModal(True)
        self.resize(400, 200)
        self.setFixedSize(400, 200)  # Запрещаем изменение размера и разворот
        self.loaded_model = None
        self.models_folder = None

        main_layout = QVBoxLayout()

        self.fps_checkbox = QCheckBox("Отображать FPS в видео")
        main_layout.addWidget(self.fps_checkbox)

        # Выбор папки с моделями
        folder_layout = QHBoxLayout()
        self.btn_select_models_folder = QPushButton("Выбрать папку с моделями")
        self.btn_select_models_folder.setIcon(QIcon("img/folder.png"))
        self.btn_select_models_folder.clicked.connect(self.select_models_folder)
        self.label_folder = QLabel("Папка не выбрана")
        folder_layout.addWidget(self.btn_select_models_folder)
        folder_layout.addWidget(self.label_folder)
        main_layout.addLayout(folder_layout)

        # Выпадающий список с локальными моделями
        self.local_model_combo = QComboBox()
        self.local_model_combo.addItem("Нет моделей")
        main_layout.addWidget(self.local_model_combo)

        # Выбор устройства
        gpu_layout = QHBoxLayout()
        gpu_layout.addWidget(QLabel("Устройство:"))
        self.gpu_combo = QComboBox()
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            for i in range(n):
                device_name = torch.cuda.get_device_name(i)
                self.gpu_combo.addItem(f"GPU {i}: {device_name}", f"cuda:{i}")
        else:
            self.gpu_combo.addItem("CPU", "cpu")
        gpu_layout.addWidget(self.gpu_combo)
        main_layout.addLayout(gpu_layout)

        # Кнопки: Загрузить локальную модель и Скачать модели
        btns_layout = QHBoxLayout()
        self.btn_load_local_model = QPushButton("Загрузить локальную модель")
        self.btn_load_local_model.setIcon(QIcon("img/start.png"))
        self.btn_load_local_model.clicked.connect(self.load_local_model)
        self.btn_download_models = QPushButton("Скачать модели")
        self.btn_download_models.setIcon(QIcon("img/download.png"))
        self.btn_download_models.clicked.connect(self.open_download_manager)
        btns_layout.addWidget(self.btn_load_local_model)
        btns_layout.addWidget(self.btn_download_models)
        main_layout.addLayout(btns_layout)

        # Кнопка Отмена
        cancel_layout = QHBoxLayout()
        self.cancel_button = QPushButton("Отмена")
        self.cancel_button.clicked.connect(self.reject)
        cancel_layout.addWidget(self.cancel_button)
        main_layout.addLayout(cancel_layout)

        self.setLayout(main_layout)

    def select_models_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с моделями")
        if folder:
            self.models_folder = folder
            self.label_folder.setText(folder)
            files = [f for f in os.listdir(folder) if f.endswith('.pt')]
            self.local_model_combo.clear()
            if files:
                for file in files:
                    self.local_model_combo.addItem(file)
            else:
                self.local_model_combo.addItem("Нет моделей")

    def load_local_model(self):
        if not self.models_folder:
            QMessageBox.warning(self, "Ошибка", "Сначала выберите папку с моделями!")
            return
        selected_model = self.local_model_combo.currentText()
        if selected_model in ["Нет моделей"]:
            QMessageBox.warning(self, "Ошибка", "Нет выбранной модели!")
            return
        model_path = os.path.join(self.models_folder, selected_model)
        device = self.gpu_combo.currentData()
        self.loader_thread = ModelLoaderThread(model_path, device)
        self.loader_thread.finished_signal.connect(self.model_loaded)
        self.loader_thread.error_signal.connect(self.model_load_error)
        self.loader_thread.start()

    def model_loaded(self, model):
        self.loaded_model = model
        QMessageBox.information(self, "Успех",
                                f"Модель {os.path.basename(self.loader_thread.model_file)} успешно загружена.")
        self.accept()

    def model_load_error(self, error_message):
        QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить модель:\n{error_message}")

    def open_download_manager(self):
        # Создаём окно загрузчика моделей без родительского окна
        # и устанавливаем модальность и флаг "всегда поверх"
        self.download_manager = DownloadManagerWindow()
        self.download_manager.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.download_manager.setWindowFlags(self.download_manager.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.download_manager.show()
        self.download_manager.raise_()
        self.download_manager.activateWindow()

    def selected_model(self):
        return self.loaded_model


# --- SplashScreen ---
class SplashScreen(QWidget):
    """
    Экран загрузки с GIF-анимацией (файл loading.gif должен быть в папке img).
    """
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setStyleSheet("background-color: black; color: white;")
        layout = QVBoxLayout()
        self.label = QLabel("Загрузка, пожалуйста, подождите...")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)
        self.movie_label = QLabel()
        self.movie_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.movie = QMovie("img/loading.gif")
        self.movie.setScaledSize(QSize(65, 50))
        self.movie_label.setMovie(self.movie)
        self.movie.start()
        layout.addWidget(self.movie_label)
        self.setLayout(layout)
        self.resize(300, 150)
        screen_geom = QApplication.primaryScreen().availableGeometry()
        self.move(screen_geom.center() - self.rect().center())


# --- VideoPlayerWindow ---
class VideoPlayerWindow(QMainWindow):
    """
    Главное окно приложения – видеоплеер с детекцией.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EzLook YoLo detector")
        self.resize(800, 600)
        self.setFixedSize(800, 600)  # Запрещаем растягивать и разворачивать основное окно
        self.selected_camera_index = None
        self.show_fps = False  # По умолчанию не показываем FPS

        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet("color: white; background-color: #333; padding: 5px;")

        # Панель управления источником видео
        self.control_panel = QWidget()
        control_layout = QHBoxLayout()
        self.radio_webcam = QRadioButton("Веб-камера")
        self.radio_file = QRadioButton("Видео файл")
        self.radio_webcam.setChecked(True)
        control_layout.addWidget(self.radio_webcam)
        control_layout.addWidget(self.radio_file)
        self.btn_camera_select = QPushButton("Выбрать камеру")
        self.btn_camera_select.setIcon(QIcon("img/camera.png"))
        self.btn_camera_select.clicked.connect(self.open_camera_selection)
        control_layout.addWidget(self.btn_camera_select)
        self.load_button = QPushButton("Загрузить")
        self.load_button.setIcon(QIcon("img/start.png"))
        control_layout.addWidget(self.load_button)
        self.version_settings_button = QToolButton()
        self.version_settings_button.setIcon(QIcon("img/settings.png"))
        self.version_settings_button.clicked.connect(self.open_settings)
        control_layout.addWidget(self.version_settings_button)
        self.control_panel.setLayout(control_layout)
        self.radio_webcam.toggled.connect(self.toggle_source)
        self.radio_file.toggled.connect(self.toggle_source)

        # Контролы для выбора файла видео
        self.file_path_input = QLineEdit()
        self.file_path_input.setReadOnly(True)
        self.file_browse_button = QToolButton()
        self.file_browse_button.setIcon(QIcon("img/folder.png"))
        self.file_browse_button.clicked.connect(self.browse_file)
        self.file_controls = QWidget()
        file_controls_layout = QHBoxLayout()
        file_controls_layout.addWidget(self.file_path_input)
        file_controls_layout.addWidget(self.file_browse_button)
        self.file_controls.setLayout(file_controls_layout)
        self.file_controls.setVisible(False)

        self.video_label = QLabel("Видео поток будет отображаться здесь")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_controls = QWidget()
        video_controls_layout = QHBoxLayout()
        self.play_pause_button = QToolButton()
        self.play_pause_button.setFixedSize(60, 60)
        self.play_pause_button.setStyleSheet("background: transparent; border: none;")
        self.play_pause_button.setIcon(QIcon("img/play.png"))
        self.video_slider = ClickableSlider(Qt.Orientation.Horizontal)
        self.video_slider.setMinimum(0)
        self.video_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.video_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 12px;
                background: #bbb;
                border-radius: 6px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                border: none;
                width: 20px;
                height: 20px;
                margin: -5px 0;
                border-radius: 10px;
            }
        """)
        self.fullscreen_button = QPushButton()
        self.fullscreen_button.setIcon(QIcon("img/fullscreen.png"))
        self.fullscreen_button.setFixedSize(60, 60)
        self.fullscreen_button.setStyleSheet("background: transparent; border: none;")
        video_controls_layout.addWidget(self.play_pause_button)
        video_controls_layout.addWidget(self.video_slider)
        video_controls_layout.addWidget(self.fullscreen_button)
        self.video_controls.setLayout(video_controls_layout)
        self.video_controls.setVisible(True)

        # Формирование основного макета
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.info_label)
        main_layout.addWidget(self.control_panel)
        main_layout.addWidget(self.file_controls)
        main_layout.addWidget(self.video_label, stretch=1)
        main_layout.addWidget(self.video_controls)
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Подключение сигналов
        self.load_button.clicked.connect(self.start_detection)
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        self.video_slider.sliderPressed.connect(self.slider_pressed)
        self.video_slider.sliderReleased.connect(self.slider_released)
        self.fullscreen_button.clicked.connect(self.enter_fullscreen)
        self.fullscreen_window = None
        self.fullscreen_slider_being_moved = False

        # Инициализация модели и параметров
        self.model = None
        self.current_model_version = "YOLOv8 Small"
        self.video_thread = None
        self.selected_file = ""
        self.selected_camera_index = None

        self.update_info_label()

    def closeEvent(self, event):
        # Если поток видео запущен – остановить его
        if self.video_thread is not None:
            self.video_thread.stop()
            self.video_thread = None
        # Если открыт полноэкранный режим – закрыть его
        if self.fullscreen_window:
            self.fullscreen_window.close()
            self.fullscreen_window = None
        event.accept()  # Допускаем закрытие окна

    def update_info_label(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.info_label.setText(f"Модель: {self.current_model_version} | Устройство: {device}")

    def open_camera_selection(self):
        dlg = CameraSelectionDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            cam_index = dlg.selected_camera()
            if cam_index is not None:
                self.selected_camera_index = cam_index
                QMessageBox.information(self, "Камера выбрана", f"Выбрана камера {cam_index}")

    def toggle_source(self):
        if self.video_thread is not None:
            self.video_thread.stop()
            self.video_thread = None
        if self.radio_webcam.isChecked():
            self.btn_camera_select.setVisible(True)
            self.file_controls.setVisible(False)
            self.video_slider.setVisible(False)
        else:
            self.btn_camera_select.setVisible(False)
            self.file_controls.setVisible(True)
            self.video_slider.setVisible(True)
        self.video_label.clear()
        self.update_info_label()

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выбрать видео файл", "", "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv)"
        )
        if file_path:
            self.selected_file = file_path
            self.file_path_input.setText(os.path.basename(file_path))

    def open_settings(self):
        dlg = SettingsDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            loaded_model = dlg.selected_model()
            if loaded_model is not None:
                self.model = loaded_model
                self.current_model_version = os.path.basename(dlg.loader_thread.model_file)
                self.show_fps = dlg.fps_checkbox.isChecked()  # сохраняем настройку отображения FPS
                self.update_info_label()
            else:
                QMessageBox.warning(self, "Предупреждение", "Не удалось загрузить выбранную модель.")

    def start_detection(self):
        if self.video_thread is not None:
            self.video_thread.stop()
            self.video_thread = None
        if self.radio_webcam.isChecked():
            if self.selected_camera_index is None:
                QMessageBox.warning(self, "Предупреждение", "Сначала выберите веб-камеру!")
                return
            cap = cv2.VideoCapture(int(self.selected_camera_index))
            is_file = False
        else:
            if not self.selected_file or not os.path.exists(self.selected_file):
                QMessageBox.warning(self, "Предупреждение", "Файл не найден!")
                return
            cap = cv2.VideoCapture(self.selected_file)
            is_file = True
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                self.video_slider.setMaximum(total_frames)
                self.video_slider.setEnabled(True)
            else:
                self.video_slider.setEnabled(False)
        if not cap.isOpened():
            QMessageBox.warning(self, "Предупреждение", "Не удалось открыть источник видео!")
            return
        if self.model is None:
            QMessageBox.critical(self, "Ошибка",
                                 "Модель не загружена. Проверьте интернет-соединение или выберите другую версию в настройках.")
            return
        self.video_thread = VideoThread(cap, self.model, is_file=is_file, show_fps=self.show_fps)
        self.video_thread.frame_signal.connect(self.update_frame)
        if is_file:
            self.video_thread.position_signal.connect(self.update_slider)
        self.video_thread.start()
        self.play_pause_button.setIcon(QIcon("img/pause.png"))
        if self.fullscreen_window:
            self.fullscreen_window.play_pause_button.setIcon(QIcon("img/pause.png"))
        self.update_info_label()

    def update_frame(self, cv_img):
        qt_img_main = self.convert_cv_qt(cv_img, self.video_label.width(), self.video_label.height(),
                                         Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(qt_img_main)
        if self.fullscreen_window:
            qt_img_full = self.convert_cv_qt(cv_img, self.fullscreen_window.video_label.width(),
                                             self.fullscreen_window.video_label.height(),
                                             Qt.AspectRatioMode.KeepAspectRatio)
            self.fullscreen_window.video_label.setPixmap(qt_img_full)

    def update_slider(self, current_frame, total_frames):
        if not hasattr(self, 'slider_being_moved'):
            self.slider_being_moved = False
        if not self.slider_being_moved:
            self.video_slider.setMaximum(total_frames)
            self.video_slider.setValue(current_frame)
        if self.fullscreen_window and not self.fullscreen_slider_being_moved:
            self.fullscreen_window.video_slider.setMaximum(total_frames)
            self.fullscreen_window.video_slider.setValue(current_frame)

    def slider_pressed(self):
        self.slider_being_moved = True

    def slider_released(self):
        self.slider_being_moved = False
        seek_val = self.video_slider.value()
        if self.video_thread is not None:
            self.video_thread.seek_position = seek_val

    def fullscreen_slider_pressed(self):
        self.fullscreen_slider_being_moved = True

    def fullscreen_slider_released(self):
        self.fullscreen_slider_being_moved = False
        seek_val = self.fullscreen_window.video_slider.value()
        if self.video_thread is not None:
            self.video_thread.seek_position = seek_val

    def convert_cv_qt(self, cv_img, target_width, target_height, keep_ratio=Qt.AspectRatioMode.KeepAspectRatio):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        scaled_image = qt_image.scaled(target_width, target_height, keep_ratio,
                                       Qt.TransformationMode.SmoothTransformation)
        return QPixmap.fromImage(scaled_image)

    def toggle_play_pause(self):
        if self.video_thread:
            self.video_thread.toggle_pause()
            if self.video_thread._paused:
                self.play_pause_button.setIcon(QIcon("img/play.png"))
                if self.fullscreen_window:
                    self.fullscreen_window.play_pause_button.setIcon(QIcon("img/play.png"))
            else:
                self.play_pause_button.setIcon(QIcon("img/pause.png"))
                if self.fullscreen_window:
                    self.fullscreen_window.play_pause_button.setIcon(QIcon("img/pause.png"))

    def enter_fullscreen(self):
        if self.fullscreen_window is None:
            self.fullscreen_window = FullscreenWindow(self)
            self.fullscreen_window.video_slider.setMaximum(self.video_slider.maximum())
            self.fullscreen_window.video_slider.setValue(self.video_slider.value())
            self.fullscreen_window.video_slider.sliderPressed.connect(self.fullscreen_slider_pressed)
            self.fullscreen_window.video_slider.sliderReleased.connect(self.fullscreen_slider_released)
            self.fullscreen_window.play_pause_button.clicked.connect(self.toggle_play_pause)
            self.fullscreen_window.exit_button.clicked.connect(self.exit_fullscreen)
            self.fullscreen_window.showFullScreen()

    def exit_fullscreen(self):
        if self.fullscreen_window:
            self.fullscreen_window.close()
            self.fullscreen_window = None


# ===============================
# Классы из downloading.py
# ===============================

YOLO_MODELS = {
    '5': ['n', 's', 'm', 'l', 'x'],
    '8': ['n', 's', 'm', 'l', 'x'],
    '11': ['n', 's', 'm', 'l', 'x']
}


class DownloadThread(QThread):
    progress = pyqtSignal(str)

    def __init__(self, model_version, model_variant, save_dir):
        super().__init__()
        self.model_version = model_version
        self.model_variant = model_variant
        self.save_dir = save_dir

    def run(self):
        try:
            if self.model_version == '11':
                model_name = f'yolo{self.model_version}{self.model_variant}'
            else:
                model_name = f'yolov{self.model_version}{self.model_variant}'
            self.progress.emit(f'Загрузка модели {model_name}...')
            model = YOLO(f'{model_name}.pt')
            os.makedirs(self.save_dir, exist_ok=True)
            model_path = os.path.join(self.save_dir, f'{model_name}.pt')
            model.save(model_path)
            self.progress.emit(f'Модель {model_name} успешно сохранена в {model_path}')
        except Exception as e:
            self.progress.emit(f'Ошибка при загрузке модели: {str(e)}')


class DownloadManagerWindow(QMainWindow):
    def __init__(self):
        super().__init__(None)
        # Устанавливаем модальность и флаг "всегда поверх"
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowTitle('Загрузчик моделей YOLO | EzLook YoLo detector')
        self.setGeometry(300, 300, 400, 200)
        self.setFixedSize(400, 200)  # Фиксированный размер окна
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.version_label = QLabel('Выберите версию YOLO:')
        self.version_combo = QComboBox()
        self.version_combo.addItems(YOLO_MODELS.keys())
        self.version_combo.currentTextChanged.connect(self.update_variants)

        self.variant_label = QLabel('Выберите вариант модели:')
        self.variant_combo = QComboBox()

        self.dir_label = QLabel('Директория для сохранения:')
        self.dir_button = QPushButton('Выбрать директорию')
        self.dir_button.clicked.connect(self.select_directory)
        self.selected_dir = QLabel('Не выбрано')

        self.download_button = QPushButton('Загрузить модель')
        self.download_button.clicked.connect(self.download_model)

        self.loading_label = QLabel()
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_movie = QMovie(os.path.join('img', 'loading.gif'))
        self.loading_label.setMovie(self.loading_movie)
        self.loading_label.setVisible(False)

        layout.addWidget(self.version_label)
        layout.addWidget(self.version_combo)
        layout.addWidget(self.variant_label)
        layout.addWidget(self.variant_combo)
        layout.addWidget(self.dir_label)
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(self.dir_button)
        dir_layout.addWidget(self.selected_dir)
        layout.addLayout(dir_layout)
        layout.addWidget(self.download_button)
        layout.addWidget(self.loading_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.update_variants(self.version_combo.currentText())

    def update_variants(self, version):
        self.variant_combo.clear()
        variants = YOLO_MODELS.get(version, [])
        self.variant_combo.addItems(variants)

    def select_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, 'Выберите директорию')
        if dir_path:
            self.selected_dir.setText(dir_path)

    def download_model(self):
        version = self.version_combo.currentText()
        variant = self.variant_combo.currentText()
        save_dir = self.selected_dir.text()

        if not save_dir or save_dir == 'Не выбрано':
            QMessageBox.warning(self, 'Ошибка', 'Пожалуйста, выберите директорию для сохранения.')
            return

        self.loading_label.setVisible(True)
        self.loading_movie.start()
        self.download_button.setEnabled(False)

        self.thread = DownloadThread(version, variant, save_dir)
        self.thread.progress.connect(self.show_message)
        self.thread.finished.connect(self.download_finished)
        self.thread.start()

    def show_message(self, message):
        QMessageBox.information(self, 'Статус загрузки', message)

    def download_finished(self):
        self.loading_movie.stop()
        self.loading_label.setVisible(False)
        self.download_button.setEnabled(True)


# ===============================
# Основной запуск приложения
# ===============================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Установка иконки для приложения (файл img/icon.ico должен находиться рядом)
    app.setWindowIcon(QIcon("img/icon.ico"))

    splash = SplashScreen()
    splash.show()
    start_time = time.time()

    window = VideoPlayerWindow()

    default_model_file = YOLO_VERSIONS["YOLOv8 Small"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_thread = ModelLoaderThread(default_model_file, device)

    def on_model_loaded(model):
        window.model = model
        window.current_model_version = "YOLOv8 Small"
        window.update_info_label()
        elapsed = time.time() - start_time
        remaining = max(0, 4 - elapsed)
        QTimer.singleShot(int(remaining * 1000), finish_splash)

    def finish_splash():
        splash.close()
        window.show()

    loader_thread.finished_signal.connect(on_model_loaded)
    loader_thread.error_signal.connect(lambda err: QMessageBox.critical(None, "Ошибка", err))
    loader_thread.start()

    sys.exit(app.exec())
