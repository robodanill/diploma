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
    Predictor,
    create_predictor,
)

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
        apply_correctness_property(preview, correctness)
        preview.setPixmap(load_pixmap(prediction.image_path, 180, 130))

        title = QLabel(prediction.image_path.name)
        title.setWordWrap(True)
        title.setObjectName("cardTitle")

        top = prediction.top_label
        if top is None:
            result_text = prediction.error or "No prediction"
        else:
            result_text = f"{top.display_name}\n{top.score:.3f}"
        if correctness is not None and not correctness.is_correct:
            result_text += f"\nCorrect: {correctness.ground_truth.display_name}"
        result = QLabel(result_text)
        result.setWordWrap(True)
        result.setObjectName("cardResult")
        apply_correctness_property(result, correctness)

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
        self.annotation_cache: dict[Path, GroundTruthLabel | None] = {}

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

        self.check_correctness_action = QAction("Check Correctness", self)
        self.check_correctness_action.setCheckable(True)
        self.check_correctness_action.toggled.connect(self._refresh_predictions)

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
        toolbar.addAction(self.check_correctness_action)
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
            QFrame#resultCard[correctness="correct"] { border: 2px solid #1f9d55; }
            QFrame#resultCard[correctness="incorrect"] { border: 2px solid #c92a2a; }
            QLabel#cardPreview {
                background: #ffffff;
                border: 3px solid transparent;
                border-radius: 6px;
            }
            QLabel#cardPreview[correctness="correct"] { border: 3px solid #1f9d55; }
            QLabel#cardPreview[correctness="incorrect"] { border: 3px solid #c92a2a; }
            QLabel#cardTitle { font-weight: 600; color: #252a31; }
            QLabel#cardResult { color: #1f7a5a; }
            QLabel#cardResult[correctness="incorrect"] { color: #a61e22; }
            QLabel#detailPreview {
                background: #ffffff;
                border: 1px solid #d9dde5;
                border-radius: 8px;
            }
            QLabel#detailPreview[correctness="correct"] { border: 4px solid #1f9d55; }
            QLabel#detailPreview[correctness="incorrect"] { border: 4px solid #c92a2a; }
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
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select folder with plant images",
            str(Path.home()),
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
        backbone, selected = QInputDialog.getItem(
            self,
            "Select model backbone",
            "Backbone:",
            list(SUPPORTED_BACKBONES),
            0,
            False,
        )
        if not selected:
            return

        try:
            self.predictor = create_predictor(
                ModelArtifacts(
                    genus_checkpoint=genus_checkpoint,
                    species_checkpoint=species_checkpoint,
                    reference_index=reference_index,
                    backbone=backbone,
                )
            )
        except Exception as exc:
            QMessageBox.critical(self, "Model load failed", str(exc))
            self.statusBar().showMessage("Model load failed")
            return

        self.statusBar().showMessage(f"Loaded trained S-CNN artifacts ({backbone})")

    def clear_all(self) -> None:
        self.image_paths.clear()
        self.predictions.clear()
        self.annotation_cache.clear()
        self.file_list.clear()
        self.detail_image_select.clear()
        self.detail_preview.clear()
        apply_correctness_property(self.detail_preview, None, repolish=True)
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
        correctness = self._prediction_correctness(prediction)
        apply_correctness_property(self.detail_preview, correctness, repolish=True)
        self.detail_preview.setPixmap(load_pixmap(prediction.image_path, 760, 420))
        lines = [f"File: {prediction.image_path}", ""]
        if prediction.error:
            lines.append(f"Error: {prediction.error}")
        else:
            lines.append("Top predictions:")
            for rank, label in enumerate(prediction.labels, start=1):
                lines.append(
                    f"{rank}. {label.display_name} | "
                    f"family: {label.family} | score: {label.score:.3f}"
                )
        if correctness is not None:
            correctness_text = "correct" if correctness.is_correct else "incorrect"
            lines.extend(["", f"Correctness: {correctness_text}"])
            if not correctness.is_correct:
                correct_answer = correctness.ground_truth.display_name
                if correctness.ground_truth.family:
                    correct_answer += f" | family: {correctness.ground_truth.family}"
                lines.append(f"Correct answer: {correct_answer}")
        self.detail_text.setPlainText("\n".join(lines))

    def _refresh_predictions(self) -> None:
        self._render_grid()
        self._render_selected_detail(self.detail_image_select.currentIndex())

    def _prediction_correctness(self, prediction: ImagePrediction) -> PredictionCorrectness | None:
        if not self.check_correctness_action.isChecked():
            return None
        return evaluate_prediction(prediction, self._ground_truth_label(prediction.image_path))

    def _ground_truth_label(self, image_path: Path) -> GroundTruthLabel | None:
        key = image_path.resolve()
        if key not in self.annotation_cache:
            self.annotation_cache[key] = load_ground_truth_label(image_path)
        return self.annotation_cache[key]

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
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
