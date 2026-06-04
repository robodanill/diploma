from __future__ import annotations

import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from PyQt6.QtCore import QObject, QRect, QRunnable, Qt, QThreadPool, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction, QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from plant_classifier.app.annotations import (
    GroundTruthLabel,
    PredictionCorrectness,
    evaluate_prediction,
    load_ground_truth_label,
)
from plant_classifier.inference import (
    SUPPORTED_BACKBONES,
    ImagePrediction,
    ModelArtifacts,
    PredictionLabel,
    Predictor,
    create_predictor,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WEIGHTS_DIR = PROJECT_ROOT / "weights"


class PredictionSignals(QObject):
    finished = pyqtSignal(list)
    failed = pyqtSignal(str)


class PredictionTask(QRunnable):
    def __init__(self, predictor: Predictor, image_paths: list[Path], top_k: int = 5) -> None:
        super().__init__()
        self.predictor = predictor
        self.image_paths = image_paths
        self.top_k = top_k
        self.signals = PredictionSignals()

    @pyqtSlot()
    def run(self) -> None:
        try:
            self.signals.finished.emit(self.predictor.predict_many(self.image_paths, self.top_k))
        except Exception as exc:  # pragma: no cover - defensive UI boundary
            self.signals.failed.emit(str(exc))


class DetailPreview(QLabel):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap = QPixmap()
        self._correctness: PredictionCorrectness | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def set_image(
        self,
        image_path: Path,
        correctness: PredictionCorrectness | None = None,
    ) -> None:
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            pixmap = QPixmap(760, 420)
            pixmap.fill(Qt.GlobalColor.lightGray)
        self._pixmap = pixmap
        self._correctness = correctness
        self.update()

    def clear(self) -> None:
        self._pixmap = QPixmap()
        self._correctness = None
        super().clear()
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ANN001
        super().paintEvent(event)
        if self._pixmap.isNull():
            return

        available_rect = self.contentsRect().adjusted(12, 12, -12, -12)
        scaled_size = self._pixmap.size()
        scaled_size.scale(
            available_rect.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        image_rect = QRect(
            available_rect.x() + (available_rect.width() - scaled_size.width()) // 2,
            available_rect.y() + (available_rect.height() - scaled_size.height()) // 2,
            scaled_size.width(),
            scaled_size.height(),
        )

        painter = QPainter(self)
        painter.drawPixmap(image_rect, self._pixmap)
        if self._correctness is None:
            painter.end()
            return

        color = "#1f9d55" if self._correctness.is_correct else "#c92a2a"
        painter.setPen(QPen(QColor(color), 5))
        painter.drawRect(image_rect.adjusted(2, 2, -2, -2))
        painter.end()


class ResultCard(QFrame):
    def __init__(
        self,
        prediction: ImagePrediction,
        correctness: PredictionCorrectness | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("resultCard")
        apply_correctness_property(self, correctness)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumWidth(220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        preview = QLabel()
        preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview.setFixedSize(180, 130)
        preview.setObjectName("cardPreview")
        preview.setPixmap(load_pixmap(prediction.image_path, 180, 130))

        title = QLabel(display_name(prediction.image_path))
        title.setWordWrap(True)
        title.setObjectName("cardTitle")

        top = prediction.top_label
        if top is None:
            result_text = prediction.error or ""
        else:
            result_text = f"{top.display_name}\nВероятность: {top.score:.3f}"
        if correctness is not None and not correctness.is_correct:
            result_text += f"\nПравильно: {correctness.ground_truth.display_name}"
        result = QLabel(result_text)
        result.setWordWrap(True)
        result.setObjectName("cardResult")
        apply_correctness_property(result, correctness)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.addWidget(preview, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(title)
        layout.addWidget(result)


class ResultsDialog(QDialog):
    def __init__(
        self,
        predictions: list[ImagePrediction],
        correctness_by_prediction: dict[Path, PredictionCorrectness],
        reference_images: dict[Path, list[Path]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Результаты")
        self.resize(980, 720)

        correct_count = sum(
            1 for correctness in correctness_by_prediction.values() if correctness.is_correct
        )
        accuracy = correct_count / len(predictions) * 100 if predictions else 0.0
        summary = QLabel(
            f"Количество изображений: {len(predictions)}\n"
            f"Количество правильных видов: {correct_count}\n"
            f"Точность: {accuracy:.2f}%"
        )
        summary.setObjectName("resultsSummary")

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(14)
        content_layout.setContentsMargins(12, 12, 12, 12)

        ordered_predictions = sorted(
            (
                prediction
                for prediction in predictions
                if prediction.image_path.resolve() in correctness_by_prediction
            ),
            key=lambda prediction: correctness_by_prediction[
                prediction.image_path.resolve()
            ].is_correct,
        )
        if not ordered_predictions:
            empty = QLabel("Нет результатов с разметкой для проверки.")
            empty.setObjectName("emptyResults")
            content_layout.addWidget(empty)
        else:
            for prediction in ordered_predictions:
                content_layout.addWidget(
                    self._prediction_row(
                        prediction,
                        correctness_by_prediction[prediction.image_path.resolve()],
                        reference_images.get(prediction.image_path.resolve(), []),
                    )
                )

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)

        close_button = QPushButton("Закрыть")
        close_button.clicked.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(summary)
        layout.addWidget(scroll, stretch=1)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)
        self.setStyleSheet(
            """
            QDialog { background: #f3f5f2; color: #202723; }
            QLabel#resultsSummary {
                background: #ffffff;
                border: 1px solid #d6ded7;
                border-radius: 8px;
                padding: 12px;
                font-weight: 700;
            }
            QFrame#predictionResult {
                background: #ffffff;
                border: 1px solid #d6ded7;
                border-radius: 8px;
                padding: 10px;
            }
            QFrame#predictionResult[correctness="correct"] { border: 2px solid #1f9d55; }
            QFrame#predictionResult[correctness="incorrect"] { border: 2px solid #c92a2a; }
            QLabel#wrongTitle { font-weight: 700; color: #252a31; }
            QLabel#wrongCaption { color: #435147; }
            QPushButton {
                background: #236b4c;
                color: white;
                border: 0;
                border-radius: 6px;
                padding: 8px 14px;
                font-weight: 600;
            }
            QPushButton:hover { background: #1d5b40; }
            """
        )

    def _prediction_row(
        self,
        prediction: ImagePrediction,
        correctness: PredictionCorrectness,
        reference_images: list[Path],
    ) -> QWidget:
        row = QFrame()
        row.setObjectName("predictionResult")
        apply_correctness_property(row, correctness)

        source_column = self._image_column(
            prediction.image_path,
            "Изображение",
            display_name_with_suffix(prediction.image_path),
        )

        predicted_name = prediction.top_label.display_name if prediction.top_label else "нет результата"
        title = QLabel(
            f"Правильный вид: {correctness.ground_truth.display_name}\n"
            f"Предсказанный вид: {predicted_name}"
        )
        title.setObjectName("wrongTitle")

        images_layout = QHBoxLayout()
        images_layout.setSpacing(36)
        images_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        images_layout.addWidget(source_column, stretch=1, alignment=Qt.AlignmentFlag.AlignTop)
        if reference_images:
            for path in reference_images[:2]:
                images_layout.addWidget(
                    self._image_column(
                        path,
                        predicted_name,
                        display_name_with_suffix(path),
                    ),
                    stretch=1,
                    alignment=Qt.AlignmentFlag.AlignTop,
                )
        else:
            missing = QLabel(
                "Примеры предсказанного вида не найдены."
                if not correctness.is_correct
                else "Предсказание совпало с правильным видом."
            )
            missing.setObjectName("wrongCaption")
            images_layout.addWidget(missing, stretch=2, alignment=Qt.AlignmentFlag.AlignTop)

        layout = QVBoxLayout(row)
        layout.setSpacing(0)
        layout.addWidget(title)
        layout.addSpacing(18)
        layout.addLayout(images_layout)
        return row

    def _image_column(self, path: Path, title: str, file_name: str) -> QWidget:
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        preview = QLabel()
        preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview.setFixedHeight(155)
        preview.setMinimumWidth(200)
        preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        preview.setPixmap(load_result_pixmap(path, 220, 155))

        label = QLabel(f"{title}\n{file_name}")
        label.setObjectName("wrongCaption")
        label.setWordWrap(True)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(preview, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(label)
        layout.addStretch()
        return container

class MainWindow(QMainWindow):
    def __init__(self, predictor: Predictor | None = None) -> None:
        super().__init__()
        self.predictor = predictor or create_predictor()
        self.thread_pool = QThreadPool.globalInstance()
        self.image_paths: list[Path] = []
        self.predictions: list[ImagePrediction] = []
        self.annotation_cache: dict[Path, GroundTruthLabel | None] = {}
        self.reference_image_index: dict[str, list[Path]] | None = None
        self.results_window: ResultsDialog | None = None

        self.setWindowTitle("Классификатор растений")
        self.resize(1180, 760)
        self._build_actions()
        self._build_ui()
        self._apply_styles()

    def _build_actions(self) -> None:
        self.open_images_action = QAction("Открыть изображения", self)
        self.open_images_action.triggered.connect(self.open_images)

        self.open_folder_action = QAction("Открыть папку", self)
        self.open_folder_action.triggered.connect(self.open_folder)

        self.run_action = QAction("Распознать", self)
        self.run_action.triggered.connect(self.run_predictions)

        self.clear_action = QAction("Очистить", self)
        self.clear_action.triggered.connect(self.clear_all)

        self.results_action = QAction("Результаты", self)
        self.results_action.triggered.connect(self.show_results)
        self.results_action.setEnabled(False)

        self.load_model_action = QAction("Загрузить модель", self)
        self.load_model_action.triggered.connect(self.load_model_artifacts)

        self.about_action = QAction("О приложении", self)
        self.about_action.triggered.connect(self.show_about)

    def _build_ui(self) -> None:
        help_menu = self.menuBar().addMenu("Справка")
        help_menu.addAction(self.about_action)

        toolbar = QToolBar("Основные действия")
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        toolbar.addAction(self.open_images_action)
        toolbar.addAction(self.open_folder_action)
        toolbar.addSeparator()
        toolbar.addAction(self.load_model_action)
        toolbar.addSeparator()
        toolbar.addAction(self.run_action)
        toolbar.addAction(self.results_action)
        toolbar.addAction(self.clear_action)
        self.addToolBar(toolbar)

        self.file_list = QListWidget()
        self.file_list.setObjectName("fileList")
        self.file_list.currentRowChanged.connect(self._sync_detail_selection)

        left_panel = QWidget()
        left_panel.setObjectName("leftPanel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(18, 18, 14, 18)
        left_layout.setSpacing(12)

        self.check_correctness_button = QPushButton("Проверка ответа: выключена")
        self.check_correctness_button.setCheckable(True)
        self.check_correctness_button.setObjectName("correctnessToggle")
        self.check_correctness_button.toggled.connect(self._on_check_correctness_toggled)
        left_layout.addWidget(self.check_correctness_button)

        input_title = QLabel("Входные изображения")
        input_title.setObjectName("sectionTitle")
        left_layout.addWidget(input_title)
        self.short_names_checkbox = QCheckBox("Показывать только названия")
        self.short_names_checkbox.toggled.connect(self._refresh_file_list)
        left_layout.addWidget(self.short_names_checkbox)
        left_layout.addWidget(self.file_list)

        button_row = QHBoxLayout()
        add_images = QPushButton("Файлы")
        add_images.clicked.connect(self.open_images)
        add_folder = QPushButton("Папка")
        add_folder.clicked.connect(self.open_folder)
        button_row.addWidget(add_images)
        button_row.addWidget(add_folder)
        left_layout.addLayout(button_row)

        self.tabs = QTabWidget()
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.grid_layout.setContentsMargins(18, 18, 18, 18)
        self.grid_layout.setHorizontalSpacing(16)
        self.grid_layout.setVerticalSpacing(16)
        self.grid_scroll = QScrollArea()
        self.grid_scroll.setObjectName("gridScroll")
        self.grid_scroll.setWidgetResizable(True)
        self.grid_scroll.setWidget(self.grid_container)
        self.tabs.addTab(self.grid_scroll, "Сетка")

        self.detail_image_select = QComboBox()
        self.detail_image_select.currentIndexChanged.connect(self._render_selected_detail)
        self.detail_preview = DetailPreview()
        self.detail_preview.setMinimumHeight(360)
        self.detail_preview.setObjectName("detailPreview")
        self.genus_select_label = QLabel("Наиболее вероятные роды")
        self.genus_select_label.setObjectName("sectionTitle")
        self.genus_list = QTextEdit()
        self.genus_list.setReadOnly(True)
        self.genus_list.setMaximumHeight(150)
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)

        detail_panel = QWidget()
        detail_panel.setObjectName("detailPanel")
        detail_layout = QVBoxLayout(detail_panel)
        detail_layout.setContentsMargins(18, 18, 18, 18)
        detail_layout.setSpacing(12)
        detail_layout.addWidget(self.detail_image_select)
        detail_layout.addWidget(self.detail_preview, stretch=2)
        detail_layout.addWidget(self.genus_select_label)
        detail_layout.addWidget(self.genus_list)
        detail_layout.addWidget(self.detail_text, stretch=1)
        self.tabs.addTab(detail_panel, "Изображение")

        splitter = QSplitter()
        splitter.addWidget(left_panel)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)
        self.setCentralWidget(splitter)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Добавьте изображения для начала работы")

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background: #f3f5f2; color: #202723; }
            QMenuBar {
                background: #ffffff;
                border-bottom: 1px solid #d8ded8;
                padding: 4px 8px;
            }
            QMenuBar::item {
                padding: 7px 10px;
                border-radius: 5px;
            }
            QMenuBar::item:selected { background: #eef4ef; }
            QMenu {
                background: #ffffff;
                border: 1px solid #d8ded8;
                padding: 6px;
            }
            QMenu::item {
                padding: 7px 22px;
                border-radius: 5px;
            }
            QMenu::item:selected { background: #e8f1eb; }
            QToolBar#mainToolbar {
                background: #ffffff;
                border-bottom: 1px solid #d8ded8;
                spacing: 8px;
                padding: 8px;
            }
            QToolButton {
                color: #223029;
                background: transparent;
                border: 1px solid transparent;
                border-radius: 6px;
                padding: 7px 10px;
            }
            QToolButton:hover { background: #edf3ee; border-color: #d6e1d8; }
            QToolButton:pressed { background: #dfe9e1; }
            QSplitter::handle { background: #d8ded8; }
            QWidget#leftPanel { background: #f8faf7; border-right: 1px solid #d8ded8; }
            QLabel#sectionTitle {
                color: #1f2b24;
                font-size: 15px;
                font-weight: 700;
            }
            QTabWidget::pane {
                border: 0;
                background: #f3f5f2;
            }
            QTabBar::tab {
                background: #e6ebe5;
                color: #29352d;
                padding: 9px 15px;
                margin: 0 4px 0 0;
                border-top-left-radius: 7px;
                border-top-right-radius: 7px;
            }
            QTabBar::tab:selected { background: #ffffff; }
            QScrollArea#gridScroll { border: 0; background: #f3f5f2; }
            QListWidget, QTextEdit, QComboBox {
                background: #ffffff;
                border: 1px solid #d6ded7;
                border-radius: 6px;
                padding: 6px;
                selection-background-color: #dcebe0;
                selection-color: #17221b;
            }
            QListWidget::item { padding: 7px 6px; border-radius: 5px; }
            QListWidget::item:selected { background: #dcebe0; color: #17221b; }
            QCheckBox {
                color: #344238;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #9bac9f;
                border-radius: 4px;
                background: #ffffff;
            }
            QCheckBox::indicator:checked {
                background: #236b4c;
                border-color: #236b4c;
            }
            QPushButton {
                background: #236b4c;
                color: white;
                border: 0;
                border-radius: 6px;
                padding: 8px 12px;
                font-weight: 600;
            }
            QPushButton:hover { background: #1d5b40; }
            QPushButton#correctnessToggle {
                background: #eef2ee;
                color: #536157;
                border: 1px solid #d6ded7;
            }
            QPushButton#correctnessToggle:checked {
                background: #236b4c;
                color: #ffffff;
                border-color: #236b4c;
            }
            QFrame#resultCard {
                background: #ffffff;
                border: 1px solid #d6ded7;
                border-radius: 8px;
                padding: 10px;
            }
            QFrame#resultCard[correctness="correct"] { border: 2px solid #1f9d55; }
            QFrame#resultCard[correctness="incorrect"] { border: 2px solid #c92a2a; }
            QLabel#cardPreview {
                background: #f7f9f7;
                border: 1px solid #d6ded7;
                border-radius: 6px;
            }
            QLabel#cardTitle { font-weight: 700; color: #252a31; }
            QLabel#cardTitle { margin-top: 4px; }
            QLabel#cardResult { color: #236b4c; }
            QLabel#cardResult[correctness="incorrect"] { color: #a61e22; }
            QLabel#detailPreview {
                background: #ffffff;
                border: 1px solid #d6ded7;
                border-radius: 8px;
            }
            QStatusBar {
                background: #ffffff;
                border-top: 1px solid #d8ded8;
                color: #435147;
            }
            """
        )

    def show_about(self) -> None:
        message = QMessageBox(self)
        message.setWindowTitle("О приложении")
        message.setIcon(QMessageBox.Icon.Information)
        message.setText("Классификатор растений")
        message.setInformativeText(
            "Я сделал это приложение для распознавания растений по изображениям листьев.\n\n"
            "Программа загружает отдельные файлы или папки с изображениями, запускает модель "
            "классификации и показывает несколько наиболее вероятных вариантов с оценкой "
            "уверенности. При наличии разметки в XML или metadata.csv можно включить проверку "
            "ответа и сравнить предсказание с правильным видом.\n\n"
            "Основной сценарий работы: выбрать изображения, при необходимости загрузить "
            "артефакты обученной S-CNN модели, запустить распознавание и посмотреть результат "
            "в виде сетки или подробной карточки изображения."
        )
        message.setStandardButtons(QMessageBox.StandardButton.Ok)
        message.button(QMessageBox.StandardButton.Ok).setText("Понятно")
        message.exec()

    def open_images(self) -> None:
        dialog = QFileDialog(self, "Выберите изображения растений", str(PROJECT_ROOT))
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setNameFilter("Изображения (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)")
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        if dialog.exec():
            self._add_paths([Path(file) for file in dialog.selectedFiles()])

    def open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку с изображениями растений",
            str(PROJECT_ROOT),
        )
        if not folder:
            return
        paths = sorted(path for path in Path(folder).rglob("*") if is_image_path(path))
        self._add_paths(paths)

    def _add_paths(self, paths: list[Path]) -> None:
        existing = {path.resolve() for path in self.image_paths}
        new_paths = [
            path for path in paths if is_image_path(path) and path.resolve() not in existing
        ]
        if not new_paths:
            return
        self.image_paths.extend(new_paths)
        self.predictions.extend(ImagePrediction(image_path=path, labels=()) for path in new_paths)
        self._refresh_file_list()
        self._render_grid()
        self._render_detail_options()
        self.results_action.setEnabled(False)
        self.statusBar().showMessage(f"Загружено изображений: {len(self.image_paths)}")

    def _refresh_file_list(self) -> None:
        current_row = self.file_list.currentRow()
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for path in self.image_paths:
            self.file_list.addItem(QListWidgetItem(self._input_list_text(path)))
        self.file_list.blockSignals(False)
        if 0 <= current_row < self.file_list.count():
            self.file_list.setCurrentRow(current_row)

    def _input_list_text(self, path: Path) -> str:
        if self.short_names_checkbox.isChecked():
            return path.stem
        return display_path(path)

    def run_predictions(self) -> None:
        if not self.image_paths:
            QMessageBox.information(
                self,
                "Нет изображений",
                "Сначала выберите изображения или папку.",
            )
            return

        self.statusBar().showMessage("Выполняется распознавание...")
        self.run_action.setEnabled(False)
        task = PredictionTask(self.predictor, self.image_paths)
        task.signals.finished.connect(self._on_predictions_finished)
        task.signals.failed.connect(self._on_predictions_failed)
        self.thread_pool.start(task)

    def load_model_artifacts(self) -> None:
        QMessageBox.information(
            self,
            "Веса для родов",
            "Загрузите веса модели, которая определяет род растения.",
        )
        genus_checkpoint = self._select_artifact("Выберите веса для родов")
        if genus_checkpoint is None:
            return

        QMessageBox.information(
            self,
            "Веса для видов",
            "Загрузите веса модели, которая определяет вид растения.",
        )
        species_checkpoint = self._select_artifact("Выберите веса для видов")
        if species_checkpoint is None:
            return

        QMessageBox.information(
            self,
            "Индекс эталонов",
            "Загрузите индекс эталонных изображений для сравнения.",
        )
        reference_index = self._select_artifact("Выберите индекс эталонов")
        if reference_index is None:
            return
        backbone, selected = QInputDialog.getItem(
            self,
            "Выберите backbone модели",
            "Архитектура:",
            list(SUPPORTED_BACKBONES),
            0,
            False,
        )
        if not selected:
            return

        artifacts = ModelArtifacts(
            genus_checkpoint=genus_checkpoint,
            species_checkpoint=species_checkpoint,
            reference_index=reference_index,
            backbone=backbone,
        )
        if not self._load_model_artifacts(artifacts, show_success=True):
            return

    def try_load_default_weights(self) -> None:
        artifacts = default_weights_artifacts()
        if artifacts is None:
            return
        if self._load_model_artifacts(artifacts, show_success=False):
            QMessageBox.information(
                self,
                "Веса загружены",
                "Веса из папки weights загружены.",
            )

    def _load_model_artifacts(self, artifacts: ModelArtifacts, *, show_success: bool) -> bool:
        try:
            self.predictor = create_predictor(artifacts)
        except Exception as exc:
            QMessageBox.critical(self, "Не удалось загрузить модель", str(exc))
            self.statusBar().showMessage("Не удалось загрузить модель")
            return False

        message = f"Загружены веса модели ({artifacts.backbone})"
        self.statusBar().showMessage(message)
        if show_success:
            QMessageBox.information(self, "Модель загружена", message)
        return True

    def clear_all(self) -> None:
        self.image_paths.clear()
        self.predictions.clear()
        self.annotation_cache.clear()
        self.file_list.clear()
        self.detail_image_select.clear()
        self.detail_preview.clear()
        self.genus_list.clear()
        self.detail_text.clear()
        self._clear_grid()
        self.results_action.setEnabled(False)
        self.statusBar().showMessage("Список очищен")

    def _on_predictions_finished(self, predictions: list[ImagePrediction]) -> None:
        self.predictions = predictions
        self.run_action.setEnabled(True)
        self._render_grid()
        self._render_detail_options()
        self.results_action.setEnabled(True)
        self.statusBar().showMessage(f"Распознавание завершено, изображений: {len(predictions)}")

    def _on_predictions_failed(self, message: str) -> None:
        self.run_action.setEnabled(True)
        QMessageBox.critical(self, "Ошибка распознавания", message)
        self.statusBar().showMessage("Ошибка распознавания")

    def _render_grid(self) -> None:
        self._clear_grid()
        columns = 3
        for index, prediction in enumerate(self.predictions):
            self.grid_layout.addWidget(
                ResultCard(prediction, self._prediction_correctness(prediction)),
                index // columns,
                index % columns,
            )

    def _clear_grid(self) -> None:
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _render_detail_options(self) -> None:
        self.detail_image_select.blockSignals(True)
        self.detail_image_select.clear()
        for prediction in self.predictions:
            self.detail_image_select.addItem(display_name(prediction.image_path))
        self.detail_image_select.blockSignals(False)
        if self.predictions:
            self.detail_image_select.setCurrentIndex(0)
            self._render_selected_detail(0)

    def _sync_detail_selection(self, row: int) -> None:
        if 0 <= row < self.detail_image_select.count():
            self.detail_image_select.setCurrentIndex(row)

    def _render_selected_detail(self, index: int) -> None:
        if not (0 <= index < len(self.predictions)):
            return
        prediction = self.predictions[index]
        correctness = self._prediction_correctness(prediction)
        self.detail_preview.set_image(prediction.image_path, correctness)
        self._render_genus_options(prediction)
        lines = [f"Файл: {display_name_with_suffix(prediction.image_path)}", ""]
        if prediction.error:
            lines.append(f"Ошибка: {prediction.error}")
        elif prediction.labels:
            lines.append("Наиболее вероятные виды:")
            for rank, label in enumerate(prediction.labels, start=1):
                lines.append(
                    f"{rank}. {label.display_name} | "
                    f"вероятность: {label.score:.3f}"
                )
        if correctness is not None:
            correctness_text = "верно" if correctness.is_correct else "неверно"
            lines.extend(["", f"Проверка: {correctness_text}"])
            if not correctness.is_correct:
                correct_answer = correctness.ground_truth.display_name
                lines.append(f"Правильный ответ: {correct_answer}")
        self.detail_text.setPlainText("\n".join(lines))

    def _on_check_correctness_toggled(self, enabled: bool) -> None:
        self.check_correctness_button.setText(
            "Проверка ответа: включена" if enabled else "Проверка ответа: выключена"
        )
        self._refresh_predictions()

    def _render_genus_options(self, prediction: ImagePrediction) -> None:
        labels = prediction.genus_labels or aggregate_genus_labels(prediction.labels)
        lines = [
            f"{rank}. {label.genus} | вероятность: {label.score:.3f}"
            for rank, label in enumerate(labels[:30], start=1)
        ]
        self.genus_list.setPlainText("\n".join(lines))

    def _refresh_predictions(self) -> None:
        self._render_grid()
        self._render_selected_detail(self.detail_image_select.currentIndex())

    def _prediction_correctness(self, prediction: ImagePrediction) -> PredictionCorrectness | None:
        if not prediction.labels and prediction.error is None:
            return None
        if not self.check_correctness_button.isChecked():
            return None
        return evaluate_prediction(prediction, self._ground_truth_label(prediction.image_path))

    def show_results(self) -> None:
        if not self.predictions:
            QMessageBox.information(self, "Нет результатов", "Сначала запустите распознавание.")
            return

        correctness_by_prediction: dict[Path, PredictionCorrectness] = {}
        reference_images: dict[Path, list[Path]] = {}
        for prediction in self.predictions:
            correctness = evaluate_prediction(
                prediction,
                self._ground_truth_label(prediction.image_path),
            )
            if correctness is None:
                continue
            key = prediction.image_path.resolve()
            correctness_by_prediction[key] = correctness
            if not correctness.is_correct and prediction.top_label is not None:
                reference_images[key] = self._reference_images_for_label(
                    prediction.top_label,
                    exclude=prediction.image_path,
                )

        self.results_window = ResultsDialog(
            self.predictions,
            correctness_by_prediction,
            reference_images,
            self,
        )
        self.results_window.show()

    def _reference_images_for_label(
        self,
        label: PredictionLabel,
        *,
        exclude: Path,
    ) -> list[Path]:
        index = self._reference_image_index()
        key = canonical_binomial(label.genus, label.species)
        excluded = exclude.resolve()
        return [path for path in index.get(key, []) if path.resolve() != excluded][:2]

    def _reference_image_index(self) -> dict[str, list[Path]]:
        if self.reference_image_index is not None:
            return self.reference_image_index

        index: dict[str, list[Path]] = defaultdict(list)
        metadata_path = PROJECT_ROOT / "leafscan" / "metadata.csv"
        if not metadata_path.is_file():
            self.reference_image_index = index
            return index

        with metadata_path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                image_name = (row.get("image_path") or "").strip()
                genus = (row.get("genus") or "").strip()
                species = (row.get("species") or "").strip()
                if not image_name or not genus or not species:
                    continue
                image_path = metadata_path.parent / image_name
                if image_path.is_file():
                    index[canonical_binomial(genus, species)].append(image_path)

        self.reference_image_index = index
        return index

    def _ground_truth_label(self, image_path: Path) -> GroundTruthLabel | None:
        key = image_path.resolve()
        if key not in self.annotation_cache:
            self.annotation_cache[key] = load_ground_truth_label(image_path)
        return self.annotation_cache[key]

    def _select_artifact(self, title: str) -> Path | None:
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            title,
            str(DEFAULT_WEIGHTS_DIR if DEFAULT_WEIGHTS_DIR.is_dir() else Path.cwd()),
            "Артефакты PyTorch (*.pt *.pth);;Все файлы (*)",
        )
        return Path(file_name) if file_name else None


def is_image_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def display_path(path: Path) -> str:
    return os.path.relpath(path.resolve(), PROJECT_ROOT)


def display_name(path: Path) -> str:
    return path.stem


def display_name_with_suffix(path: Path) -> str:
    return path.name


def aggregate_genus_labels(labels: tuple[PredictionLabel, ...]) -> tuple[PredictionLabel, ...]:
    ranked: dict[str, PredictionLabel] = {}
    for label in labels:
        current = ranked.get(label.genus)
        if current is None or label.score > current.score:
            ranked[label.genus] = PredictionLabel(
                family=label.family,
                genus=label.genus,
                species="",
                score=label.score,
            )
    return tuple(sorted(ranked.values(), key=lambda label: label.score, reverse=True))


def canonical_binomial(genus: str, species: str) -> str:
    display = species if species.lower().startswith(genus.lower()) else f"{genus} {species}"
    tokens = re.sub(r"[^0-9a-z]+", " ", display.lower()).split()
    return " ".join(tokens[:2])


def default_weights_artifacts() -> ModelArtifacts | None:
    if not DEFAULT_WEIGHTS_DIR.is_dir():
        return None

    genus_checkpoint = _first_existing_weight(
        "final_scnn_genus_*.pt",
        "scnn_genus_*_best.pt",
        "scnn_genus_*.pt",
        "*genus*.pt",
    )
    species_checkpoint = _first_existing_weight(
        "final_scnn_species_*.pt",
        "scnn_species_*_best.pt",
        "scnn_species_*.pt",
        "*species*.pt",
    )
    reference_index = _first_existing_weight(
        "final_reference_index_*.pt",
        "reference_index_*.pt",
        "*reference*index*.pt",
    )
    if not (genus_checkpoint and species_checkpoint and reference_index):
        return None

    return ModelArtifacts(
        genus_checkpoint=genus_checkpoint,
        species_checkpoint=species_checkpoint,
        reference_index=reference_index,
        backbone=_infer_backbone(
            genus_checkpoint.name,
            species_checkpoint.name,
            reference_index.name,
        ),
    )


def _first_existing_weight(*patterns: str) -> Path | None:
    for pattern in patterns:
        matches = sorted(DEFAULT_WEIGHTS_DIR.glob(pattern))
        if matches:
            return matches[0]
    return None


def _infer_backbone(*names: str) -> str:
    combined = " ".join(names).lower()
    for backbone in SUPPORTED_BACKBONES:
        if backbone.lower() in combined:
            return backbone
    return "vgg16"


def load_pixmap(path: Path, width: int, height: int) -> QPixmap:
    pixmap = QPixmap(str(path))
    if pixmap.isNull():
        placeholder = QPixmap(width, height)
        placeholder.fill(Qt.GlobalColor.lightGray)
        return placeholder
    return pixmap.scaled(
        width,
        height,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def load_result_pixmap(path: Path, width: int, height: int) -> QPixmap:
    canvas = QPixmap(width, height)
    canvas.fill(Qt.GlobalColor.white)

    pixmap = QPixmap(str(path))
    if pixmap.isNull():
        canvas.fill(Qt.GlobalColor.lightGray)
        return canvas

    scaled = pixmap.scaled(
        width,
        height,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    painter = QPainter(canvas)
    painter.drawPixmap(
        (width - scaled.width()) // 2,
        (height - scaled.height()) // 2,
        scaled,
    )
    painter.end()
    return canvas


def apply_correctness_property(
    widget: QWidget,
    correctness: PredictionCorrectness | None,
    *,
    repolish: bool = False,
) -> None:
    value = ""
    if correctness is not None:
        value = "correct" if correctness.is_correct else "incorrect"
    widget.setProperty("correctness", value)
    if repolish:
        widget.style().unpolish(widget)
        widget.style().polish(widget)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    QTimer.singleShot(0, window.try_load_default_weights)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
