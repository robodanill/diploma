from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import QObject, QRunnable, Qt, QThreadPool, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
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

from plant_classifier.inference import ImagePrediction, ModelArtifacts, Predictor, create_predictor

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


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


class ResultCard(QFrame):
    def __init__(self, prediction: ImagePrediction, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("resultCard")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumWidth(220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        preview = QLabel()
        preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview.setFixedSize(180, 130)
        preview.setPixmap(load_pixmap(prediction.image_path, 180, 130))

        title = QLabel(prediction.image_path.name)
        title.setWordWrap(True)
        title.setObjectName("cardTitle")

        top = prediction.top_label
        if top is None:
            result = QLabel(prediction.error or "No prediction")
        else:
            result = QLabel(f"{top.display_name}\n{top.score:.3f}")
        result.setWordWrap(True)
        result.setObjectName("cardResult")

        layout = QVBoxLayout(self)
        layout.addWidget(preview)
        layout.addWidget(title)
        layout.addWidget(result)


class MainWindow(QMainWindow):
    def __init__(self, predictor: Predictor | None = None) -> None:
        super().__init__()
        self.predictor = predictor or create_predictor()
        self.thread_pool = QThreadPool.globalInstance()
        self.image_paths: list[Path] = []
        self.predictions: list[ImagePrediction] = []

        self.setWindowTitle("Plant Classifier")
        self.resize(1180, 760)
        self._build_actions()
        self._build_ui()
        self._apply_styles()

    def _build_actions(self) -> None:
        self.open_images_action = QAction("Open Images", self)
        self.open_images_action.triggered.connect(self.open_images)

        self.open_folder_action = QAction("Open Folder", self)
        self.open_folder_action.triggered.connect(self.open_folder)

        self.run_action = QAction("Run", self)
        self.run_action.triggered.connect(self.run_predictions)

        self.clear_action = QAction("Clear", self)
        self.clear_action.triggered.connect(self.clear_all)

        self.load_model_action = QAction("Load Model", self)
        self.load_model_action.triggered.connect(self.load_model_artifacts)

    def _build_ui(self) -> None:
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        toolbar.addAction(self.open_images_action)
        toolbar.addAction(self.open_folder_action)
        toolbar.addSeparator()
        toolbar.addAction(self.load_model_action)
        toolbar.addSeparator()
        toolbar.addAction(self.run_action)
        toolbar.addAction(self.clear_action)
        self.addToolBar(toolbar)

        self.file_list = QListWidget()
        self.file_list.currentRowChanged.connect(self._sync_detail_selection)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel("Input images"))
        left_layout.addWidget(self.file_list)

        button_row = QHBoxLayout()
        add_images = QPushButton("Images")
        add_images.clicked.connect(self.open_images)
        add_folder = QPushButton("Folder")
        add_folder.clicked.connect(self.open_folder)
        button_row.addWidget(add_images)
        button_row.addWidget(add_folder)
        left_layout.addLayout(button_row)

        self.tabs = QTabWidget()
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.grid_scroll = QScrollArea()
        self.grid_scroll.setWidgetResizable(True)
        self.grid_scroll.setWidget(self.grid_container)
        self.tabs.addTab(self.grid_scroll, "Grid")

        self.detail_image_select = QComboBox()
        self.detail_image_select.currentIndexChanged.connect(self._render_selected_detail)
        self.detail_preview = QLabel()
        self.detail_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail_preview.setMinimumHeight(360)
        self.detail_preview.setObjectName("detailPreview")
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)

        detail_panel = QWidget()
        detail_layout = QVBoxLayout(detail_panel)
        detail_layout.addWidget(self.detail_image_select)
        detail_layout.addWidget(self.detail_preview, stretch=2)
        detail_layout.addWidget(self.detail_text, stretch=1)
        self.tabs.addTab(detail_panel, "Image")

        splitter = QSplitter()
        splitter.addWidget(left_panel)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)
        self.setCentralWidget(splitter)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Load images to start")

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background: #f6f7f9; }
            QToolBar { background: #ffffff; border-bottom: 1px solid #d9dde5; spacing: 8px; }
            QListWidget, QTextEdit, QComboBox {
                background: #ffffff;
                border: 1px solid #d6dbe4;
                border-radius: 6px;
                padding: 6px;
            }
            QPushButton {
                background: #1f7a5a;
                color: white;
                border: 0;
                border-radius: 6px;
                padding: 8px 12px;
            }
            QPushButton:hover { background: #17664b; }
            QFrame#resultCard {
                background: #ffffff;
                border: 1px solid #d9dde5;
                border-radius: 8px;
                padding: 8px;
            }
            QLabel#cardTitle { font-weight: 600; color: #252a31; }
            QLabel#cardResult { color: #1f7a5a; }
            QLabel#detailPreview {
                background: #ffffff;
                border: 1px solid #d9dde5;
                border-radius: 8px;
            }
            """
        )

    def open_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select plant images",
            str(Path.home()),
            "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)",
        )
        self._add_paths([Path(file) for file in files])

    def open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select folder with plant images", str(Path.home()))
        if not folder:
            return
        paths = sorted(path for path in Path(folder).rglob("*") if is_image_path(path))
        self._add_paths(paths)

    def _add_paths(self, paths: list[Path]) -> None:
        existing = {path.resolve() for path in self.image_paths}
        new_paths = [path for path in paths if is_image_path(path) and path.resolve() not in existing]
        if not new_paths:
            return
        self.image_paths.extend(new_paths)
        for path in new_paths:
            self.file_list.addItem(QListWidgetItem(str(path)))
        self.statusBar().showMessage(f"Loaded {len(self.image_paths)} image(s)")

    def run_predictions(self) -> None:
        if not self.image_paths:
            QMessageBox.information(self, "No images", "Select images or a folder first.")
            return

        self.statusBar().showMessage("Running predictions...")
        self.run_action.setEnabled(False)
        task = PredictionTask(self.predictor, self.image_paths)
        task.signals.finished.connect(self._on_predictions_finished)
        task.signals.failed.connect(self._on_predictions_failed)
        self.thread_pool.start(task)

    def load_model_artifacts(self) -> None:
        genus_checkpoint = self._select_artifact("Select genus checkpoint")
        if genus_checkpoint is None:
            return
        species_checkpoint = self._select_artifact("Select species checkpoint")
        if species_checkpoint is None:
            return
        reference_index = self._select_artifact("Select reference index")
        if reference_index is None:
            return

        try:
            self.predictor = create_predictor(
                ModelArtifacts(
                    genus_checkpoint=genus_checkpoint,
                    species_checkpoint=species_checkpoint,
                    reference_index=reference_index,
                )
            )
        except Exception as exc:
            QMessageBox.critical(self, "Model load failed", str(exc))
            self.statusBar().showMessage("Model load failed")
            return

        self.statusBar().showMessage("Loaded trained S-CNN model artifacts")

    def clear_all(self) -> None:
        self.image_paths.clear()
        self.predictions.clear()
        self.file_list.clear()
        self.detail_image_select.clear()
        self.detail_preview.clear()
        self.detail_text.clear()
        self._clear_grid()
        self.statusBar().showMessage("Cleared")

    def _on_predictions_finished(self, predictions: list[ImagePrediction]) -> None:
        self.predictions = predictions
        self.run_action.setEnabled(True)
        self._render_grid()
        self._render_detail_options()
        self.statusBar().showMessage(f"Finished {len(predictions)} prediction(s)")

    def _on_predictions_failed(self, message: str) -> None:
        self.run_action.setEnabled(True)
        QMessageBox.critical(self, "Prediction failed", message)
        self.statusBar().showMessage("Prediction failed")

    def _render_grid(self) -> None:
        self._clear_grid()
        columns = 3
        for index, prediction in enumerate(self.predictions):
            self.grid_layout.addWidget(ResultCard(prediction), index // columns, index % columns)

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
            self.detail_image_select.addItem(prediction.image_path.name)
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
        self.detail_preview.setPixmap(load_pixmap(prediction.image_path, 760, 420))
        lines = [f"File: {prediction.image_path}", ""]
        if prediction.error:
            lines.append(f"Error: {prediction.error}")
        else:
            lines.append("Top predictions:")
            for rank, label in enumerate(prediction.labels, start=1):
                lines.append(
                    f"{rank}. {label.display_name} | family: {label.family} | score: {label.score:.3f}"
                )
        self.detail_text.setPlainText("\n".join(lines))

    def _select_artifact(self, title: str) -> Path | None:
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            title,
            str(Path.cwd()),
            "PyTorch artifacts (*.pt *.pth);;All files (*)",
        )
        return Path(file_name) if file_name else None


def is_image_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


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


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
