import os
import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QFileDialog, QMessageBox
)
from PyQt6.QtGui import QMovie
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from ultralytics import YOLO

# Словарь доступных моделей и их вариантов
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
            # Формирование имени модели
            if self.model_version == '11':
                model_name = f'yolo{self.model_version}{self.model_variant}'
            else:
                model_name = f'yolov{self.model_version}{self.model_variant}'

            self.progress.emit(f'Загрузка модели {model_name}...')

            # Загрузка модели
            model = YOLO(f'{model_name}.pt')

            # Создание директории для сохранения модели, если она не существует
            os.makedirs(self.save_dir, exist_ok=True)

            # Путь для сохранения модели
            model_path = os.path.join(self.save_dir, f'{model_name}.pt')

            # Сохранение модели
            model.save(model_path)

            self.progress.emit(f'Модель {model_name} успешно сохранена в {model_path}')
        except Exception as e:
            self.progress.emit(f'Ошибка при загрузке модели: {str(e)}')

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Загрузчик моделей YOLO | EzLook YoLo detector')
        self.setGeometry(300, 300, 400, 200)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Выпадающий список для выбора версии модели
        self.version_label = QLabel('Выберите версию YOLO:')
        self.version_combo = QComboBox()
        self.version_combo.addItems(YOLO_MODELS.keys())
        self.version_combo.currentTextChanged.connect(self.update_variants)

        # Выпадающий список для выбора варианта модели
        self.variant_label = QLabel('Выберите вариант модели:')
        self.variant_combo = QComboBox()

        # Кнопка для выбора директории сохранения
        self.dir_label = QLabel('Директория для сохранения:')
        self.dir_button = QPushButton('Выбрать директорию')
        self.dir_button.clicked.connect(self.select_directory)
        self.selected_dir = QLabel('Не выбрано')

        # Кнопка для начала загрузки
        self.download_button = QPushButton('Загрузить модель')
        self.download_button.clicked.connect(self.download_model)

        # Метка для отображения GIF загрузки
        self.loading_label = QLabel()
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_movie = QMovie(os.path.join('img', 'loading.gif'))
        self.loading_label.setMovie(self.loading_movie)
        self.loading_label.setVisible(False)

        # Добавление виджетов в макет
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

        self.setLayout(layout)

        # Инициализация вариантов модели
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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
