"""
Thorlabs CCS200 Spectrometer Viewer
------------------------------------
A PyQt5 GUI for connecting to a Thorlabs CCS-series spectrometer,
running a live acquisition, and exporting spectra to CSV.

Functional design:
  * Hardware polling runs on a background QThread so the GUI never
    freezes while waiting on the instrument.
  * All hardware calls are wrapped in try/except with user-facing
    error dialogs instead of silent console prints.
  * Buttons enable/disable themselves based on connection/acquisition
    state, so it's not possible to e.g. hit "Start" before connecting.
  * Integration time can be changed live, while a scan is running.
  * Status bar reports connection state, acquisition state, and the
    current peak wavelength/intensity.
  * Clean shutdown on window close (stops the thread, resets the
    instrument) so the process doesn't hang or leave the device busy.
  * CSV logger: separate from the one-off "Save CSV" snapshot button.
    The user picks a destination folder; while "Start Logging" is on,
    every acquired spectrum is written out as its own CSV file (e.g.
    an integration time of 1s run for 1 minute of logging produces 60
    files), using the exact same format as the manual snapshot save:
    a wavelength_nm/intensity header, one row per wavelength sample.
    The acquisition loop is paced to the current integration time
    (instead of a fixed 50ms tick) so a new file is written once per
    integration period, and changing the integration time live
    re-paces both scanning and logging together.

Visual design:
  A dark "instrument panel" look modeled on real optical-bench gear
  (spectrum analyzers, oscilloscopes) rather than a generic dark mode:
  a phosphor-amber trace glowing on a near-black scope grid, grouped
  control "cards" for Connection / Acquisition / Export, and a
  monospaced data readout for the live peak wavelength. A status dot
  pulses gently while a scan is running.
"""

import os
import sys
import csv
import logging
from datetime import datetime

import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters

from PyQt5.QtCore import QObject, QThread, pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QPushButton,
    QLabel,
    QDoubleSpinBox,
    QFileDialog,
    QMessageBox,
    QSizePolicy,
)

from thorlabs_ccs import list_spectrometers, TLCCS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("ccs_viewer")

# Fallback poll interval used only before a real integration time is known.
DEFAULT_POLL_INTERVAL_MS = 50
# Floor so a very short integration time can't spin the loop with ~0ms sleeps.
MIN_POLL_INTERVAL_MS = 5


# ----------------------------------------------------------------------
# Design tokens
# ----------------------------------------------------------------------
class Theme:
    BG_APP = "#0F1317"
    BG_PANEL = "#161B20"
    BG_PANEL_RAISED = "#1E252C"
    BG_INPUT = "#12171C"
    BORDER = "#2A323A"
    BORDER_SOFT = "#20272E"

    TEXT_PRIMARY = "#E7EBEE"
    TEXT_SECONDARY = "#7C8894"
    TEXT_DIM = "#4C5760"

    ACCENT = "#FFB454"       # phosphor amber — the live trace / primary actions
    ACCENT_DIM = "#8A6A3D"
    PEAK = "#57D9C7"         # teal — peak marker / data readout
    RUNNING = "#3ADB8A"      # green — acquisition active
    STOPPED = "#E5484D"      # red — stop / disconnect
    IDLE = "#4C5760"         # gray — idle / disconnected

    FONT_UI = "'Segoe UI', 'Inter', 'Helvetica Neue', sans-serif"
    FONT_MONO = "'JetBrains Mono', 'Cascadia Mono', 'Consolas', monospace"


def build_stylesheet() -> str:
    t = Theme
    return f"""
    QMainWindow, QWidget {{
        background-color: {t.BG_APP};
        color: {t.TEXT_PRIMARY};
        font-family: {t.FONT_UI};
        font-size: 13px;
    }}

    QGroupBox {{
        background-color: {t.BG_PANEL};
        border: 1px solid {t.BORDER};
        border-radius: 8px;
        margin-top: 14px;
        padding: 12px 10px 10px 10px;
        font-family: {t.FONT_UI};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 12px;
        top: -2px;
        padding: 0 6px;
        color: {t.TEXT_SECONDARY};
        font-size: 10.5px;
        font-weight: 600;
        background-color: {t.BG_APP};
    }}

    QPushButton {{
        background-color: {t.BG_PANEL_RAISED};
        color: {t.TEXT_PRIMARY};
        border: 1px solid {t.BORDER};
        border-radius: 6px;
        padding: 7px 14px;
        font-weight: 500;
    }}
    QPushButton:hover:!disabled {{
        border-color: {t.ACCENT_DIM};
        color: {t.ACCENT};
    }}
    QPushButton:pressed:!disabled {{
        background-color: {t.BG_INPUT};
    }}
    QPushButton:disabled {{
        color: {t.TEXT_DIM};
        border-color: {t.BORDER_SOFT};
        background-color: {t.BG_PANEL};
    }}

    QPushButton#primaryBtn:!disabled {{
        background-color: {t.ACCENT};
        color: #241A0B;
        border: 1px solid {t.ACCENT};
        font-weight: 600;
    }}
    QPushButton#primaryBtn:hover:!disabled {{
        background-color: #FFC578;
        color: #241A0B;
    }}

    QPushButton#stopBtn:!disabled {{
        border-color: {t.STOPPED};
        color: {t.STOPPED};
    }}
    QPushButton#stopBtn:hover:!disabled {{
        background-color: {t.STOPPED};
        color: {t.BG_APP};
    }}

    QLabel {{
        color: {t.TEXT_SECONDARY};
        background: transparent;
    }}
    QLabel#appTitle {{
        color: {t.TEXT_PRIMARY};
        font-size: 15px;
        font-weight: 600;
    }}
    QLabel#appSubtitle {{
        color: {t.TEXT_DIM};
        font-size: 10.5px;
        font-family: {t.FONT_MONO};
    }}
    QLabel#fieldLabel {{
        color: {t.TEXT_SECONDARY};
        font-size: 11.5px;
    }}
    QLabel#peakReadout {{
        color: {t.PEAK};
        font-family: {t.FONT_MONO};
        font-size: 12px;
        font-weight: 600;
    }}
    QLabel#statusDot {{
        border-radius: 5px;
        background-color: {t.IDLE};
    }}

    QDoubleSpinBox {{
        background-color: {t.BG_INPUT};
        border: 1px solid {t.BORDER};
        border-radius: 6px;
        padding: 5px 8px;
        color: {t.ACCENT};
        font-family: {t.FONT_MONO};
        font-weight: 600;
        min-width: 90px;
    }}
    QDoubleSpinBox:focus {{
        border-color: {t.ACCENT_DIM};
    }}
    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
        width: 14px;
        background-color: {t.BG_PANEL_RAISED};
        border-left: 1px solid {t.BORDER};
    }}

    QStatusBar {{
        background-color: {t.BG_PANEL};
        border-top: 1px solid {t.BORDER};
        color: {t.TEXT_SECONDARY};
        font-family: {t.FONT_MONO};
        font-size: 11px;
    }}
    QStatusBar::item {{
        border: none;
    }}
    QStatusBar QLabel {{
        padding: 2px 10px;
    }}
    """


class StatusDot(QLabel):
    """A small filled circle used as a state indicator."""

    def __init__(self, diameter: int = 10):
        super().__init__()
        self.setObjectName("statusDot")
        self.setFixedSize(diameter, diameter)

    def set_color(self, hex_color: str):
        self.setStyleSheet(
            f"QLabel#statusDot {{ border-radius: {self.width() // 2}px; "
            f"background-color: {hex_color}; }}"
        )


# ----------------------------------------------------------------------
# Acquisition worker
# ----------------------------------------------------------------------
class AcquisitionWorker(QObject):
    """Runs on a background thread and repeatedly polls the spectrometer.

    Keeping this off the GUI thread matters because
    get_scan_data_factory() blocks on USB/serial I/O; doing that in a
    QTimer on the main thread would stall button clicks and redraws.

    The pacing between reads is set to the current integration time
    (``poll_interval_ms``): a new spectrum only becomes available every
    integration period, so polling faster just re-reads stale data, and
    it's this same interval that drives the CSV logger's save rate.
    ``poll_interval_ms`` can be updated live (e.g. if the user changes
    the integration time while a scan is running).
    """

    spectrum_ready = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, ccs, poll_interval_ms: int = DEFAULT_POLL_INTERVAL_MS):
        super().__init__()
        self._ccs = ccs
        self._running = False
        self.poll_interval_ms = max(poll_interval_ms, MIN_POLL_INTERVAL_MS)

    def start(self):
        self._running = True
        while self._running:
            try:
                spectrum = np.array(self._ccs.get_scan_data_factory())
                self.spectrum_ready.emit(spectrum)
            except Exception as exc:  # noqa: BLE001 - surface any hardware error
                self.error.emit(str(exc))
                self._running = False
                break
            QThread.msleep(self.poll_interval_ms)

    def stop(self):
        self._running = False


# ----------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------
class CCSWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.ccs = None
        self.wavelength = None
        self.last_spectrum = None

        self.acq_thread = None
        self.acq_worker = None
        self._blink_on = True

        # CSV logger state: while active, every incoming spectrum is
        # written out as its own CSV file (same format as the manual
        # snapshot save) into a folder the user picks.
        self.log_folder = None
        self.log_row_count = 0
        self.logging_active = False

        self.setWindowTitle("Thorlabs CCS200 Viewer")
        self.resize(1280, 820)
        self.setStyleSheet(build_stylesheet())

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 14, 16, 10)
        root.setSpacing(12)

        root.addLayout(self._build_header())
        root.addLayout(self._build_control_row())
        root.addWidget(self._build_logger_row())
        root.addWidget(self._build_plot(), stretch=1)

        self._build_status_bar()
        self._wire_signals()
        self._set_ui_state(connected=False, running=False)

        # Gentle blink for the "running" indicator — the one bit of
        # motion in an otherwise still, disciplined panel.
        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._toggle_blink)
        self._blink_timer.start(650)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_header(self) -> QHBoxLayout:
        header = QHBoxLayout()
        header.setSpacing(10)

        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        title = QLabel("THORLABS CCS200")
        title.setObjectName("appTitle")
        subtitle = QLabel("SPECTROMETER · LIVE VIEWER")
        subtitle.setObjectName("appSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)

        header.addStretch(1)

        self.status_dot = StatusDot(12)
        self.status_dot.set_color(Theme.IDLE)
        self.status_text = QLabel("DISCONNECTED")
        self.status_text.setStyleSheet(
            f"color: {Theme.TEXT_SECONDARY}; font-family: {Theme.FONT_MONO}; "
            f"font-size: 11px; font-weight: 600; letter-spacing: 1px;"
        )
        header.addWidget(self.status_dot, alignment=Qt.AlignVCenter)
        header.addWidget(self.status_text, alignment=Qt.AlignVCenter)

        return header

    def _build_control_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)

        # --- Connection card ---
        conn_group = QGroupBox("CONNECTION")
        conn_layout = QHBoxLayout(conn_group)
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setObjectName("primaryBtn")
        self.disconnect_btn = QPushButton("Disconnect")
        conn_layout.addWidget(self.connect_btn)
        conn_layout.addWidget(self.disconnect_btn)

        # --- Acquisition card ---
        acq_group = QGroupBox("ACQUISITION")
        acq_layout = QHBoxLayout(acq_group)
        self.start_btn = QPushButton("▶  Start")
        self.start_btn.setObjectName("primaryBtn")
        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setObjectName("stopBtn")
        int_label = QLabel("Integration (s)")
        int_label.setObjectName("fieldLabel")
        self.int_time = QDoubleSpinBox()
        self.int_time.setDecimals(4)
        self.int_time.setRange(0.00001, 60)
        self.int_time.setValue(0.4)
        self.int_time.setSingleStep(0.01)
        acq_layout.addWidget(self.start_btn)
        acq_layout.addWidget(self.stop_btn)
        acq_layout.addSpacing(8)
        acq_layout.addWidget(int_label)
        acq_layout.addWidget(self.int_time)

        # --- Export card ---
        export_group = QGroupBox("EXPORT")
        export_layout = QHBoxLayout(export_group)
        self.save_btn = QPushButton("⬇ CSV")
        self.save_png_btn = QPushButton("⬇ PNG")
        self.autorange_btn = QPushButton("⤢ Auto Range")
        export_layout.addWidget(self.save_btn)
        export_layout.addWidget(self.save_png_btn)
        export_layout.addWidget(self.autorange_btn)

        row.addWidget(conn_group)
        row.addWidget(acq_group, stretch=1)
        row.addWidget(export_group)

        return row

    def _build_logger_row(self) -> QWidget:
        group = QGroupBox("LOGGER  ·  one CSV file per scan, same format as snapshot")
        layout = QHBoxLayout(group)

        folder_label = QLabel("Folder:")
        folder_label.setObjectName("fieldLabel")
        self.log_folder_label = QLabel("No folder selected")
        self.log_folder_label.setStyleSheet(
            f"color: {Theme.TEXT_DIM}; font-family: {Theme.FONT_MONO}; font-size: 11px;"
        )
        self.log_folder_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Preferred
        )

        self.browse_log_btn = QPushButton("Browse…")
        self.log_toggle_btn = QPushButton("●  Start Logging")
        self.log_toggle_btn.setObjectName("primaryBtn")

        self.log_status_label = QLabel("")
        self.log_status_label.setObjectName("peakReadout")

        layout.addWidget(folder_label)
        layout.addWidget(self.log_folder_label, stretch=1)
        layout.addWidget(self.browse_log_btn)
        layout.addWidget(self.log_toggle_btn)
        layout.addSpacing(6)
        layout.addWidget(self.log_status_label)

        return group

    def _build_plot(self) -> QWidget:
        wrapper = QGroupBox("SPECTRUM")
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(4, 10, 8, 8)

        self.plot = pg.PlotWidget()
        self.plot.setBackground(Theme.BG_PANEL)
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.setLabel("bottom", "Wavelength", units="nm",
                            color=Theme.TEXT_SECONDARY)
        self.plot.setLabel("left", "Intensity", color=Theme.TEXT_SECONDARY)
        for axis_name in ("bottom", "left"):
            axis = self.plot.getAxis(axis_name)
            axis.setPen(pg.mkPen(Theme.BORDER))
            axis.setTextPen(pg.mkPen(Theme.TEXT_SECONDARY))

        trace_pen = pg.mkPen(Theme.ACCENT, width=1.8)
        fill_brush = pg.mkBrush(QColor(Theme.ACCENT).lighter(100).name())
        fill_color = QColor(Theme.ACCENT)
        fill_color.setAlpha(40)

        self.curve = self.plot.plot(
            pen=trace_pen,
            fillLevel=0,
            brush=pg.mkBrush(fill_color),
        )
        self.peak_marker = self.plot.plot(
            pen=None, symbol='o',
            symbolBrush=pg.mkBrush(Theme.PEAK),
            symbolPen=pg.mkPen(Theme.BG_PANEL, width=1.5),
            symbolSize=9,
        )

        wrapper_layout.addWidget(self.plot)
        return wrapper

    def _build_status_bar(self):
        self.status_conn = QLabel("STATE: DISCONNECTED")
        self.status_acq = QLabel("ACQ: IDLE")
        self.status_peak = QLabel("PEAK: —")
        self.status_peak.setObjectName("peakReadout")
        self.statusBar().addWidget(self.status_conn)
        self.statusBar().addWidget(self.status_acq)
        self.statusBar().addPermanentWidget(self.status_peak)

    def _wire_signals(self):
        self.connect_btn.clicked.connect(self.connect_spectrometer)
        self.disconnect_btn.clicked.connect(self.disconnect_spectrometer)
        self.start_btn.clicked.connect(self.start_acquisition)
        self.stop_btn.clicked.connect(self.stop_acquisition)
        self.save_btn.clicked.connect(self.save_csv)
        self.save_png_btn.clicked.connect(self.save_png)
        self.autorange_btn.clicked.connect(lambda: self.plot.autoRange())
        self.int_time.valueChanged.connect(self.apply_integration_time)
        self.browse_log_btn.clicked.connect(self.choose_log_folder)
        self.log_toggle_btn.clicked.connect(self.toggle_logging)

    # ------------------------------------------------------------------
    # UI state management
    # ------------------------------------------------------------------
    def _set_ui_state(self, connected: bool, running: bool):
        self.connect_btn.setEnabled(not connected)
        self.disconnect_btn.setEnabled(connected and not running)
        self.start_btn.setEnabled(connected and not running)
        self.stop_btn.setEnabled(connected and running)
        self.save_btn.setEnabled(self.last_spectrum is not None)
        self.save_png_btn.setEnabled(self.last_spectrum is not None)

        self.log_toggle_btn.setEnabled(connected or self.logging_active)
        self.browse_log_btn.setEnabled(not self.logging_active)

        if running:
            dot_color, text = Theme.RUNNING, "ACQUIRING"
        elif connected:
            dot_color, text = Theme.ACCENT, "CONNECTED"
        else:
            dot_color, text = Theme.IDLE, "DISCONNECTED"

        self.status_dot.set_color(dot_color)
        self.status_text.setText(text)
        self.status_conn.setText(f"STATE: {'CONNECTED' if connected else 'DISCONNECTED'}")
        self.status_acq.setText(f"ACQ: {'RUNNING' if running else 'IDLE'}")

    def _toggle_blink(self):
        """Softly pulse the status dot while a scan is running."""
        if not (self.acq_worker is not None):
            return
        self._blink_on = not self._blink_on
        color = Theme.RUNNING if self._blink_on else Theme.ACCENT_DIM
        self.status_dot.set_color(color)

    def _show_error(self, title: str, message: str):
        log.error("%s: %s", title, message)
        QMessageBox.critical(self, title, message)

    @staticmethod
    def _restyle_button(button: QPushButton, object_name: str):
        """Swap a button's QSS object name and force it to re-polish,
        since Qt doesn't repaint style rules on setObjectName() alone."""
        button.setObjectName(object_name)
        button.style().unpolish(button)
        button.style().polish(button)

    # ------------------------------------------------------------------
    # Connection handling
    # ------------------------------------------------------------------
    def connect_spectrometer(self):
        try:
            devices = list_spectrometers()
        except Exception as exc:
            self._show_error("Device Search Failed", str(exc))
            return

        if not devices:
            self._show_error("No Spectrometer Found",
                              "No Thorlabs CCS spectrometer was detected. "
                              "Check the USB connection and try again.")
            return

        try:
            self.ccs = TLCCS(device_info=devices[0])
            self.wavelength = np.array(self.ccs.get_wavelength())
            self.apply_integration_time(self.int_time.value())
        except Exception as exc:
            self.ccs = None
            self._show_error("Connection Failed", str(exc))
            return

        log.info("Connected to spectrometer")
        self._set_ui_state(connected=True, running=False)

    def disconnect_spectrometer(self):
        if self.ccs is None:
            return
        if self.logging_active:
            self.stop_logging()
        try:
            self.ccs.reset()
        except Exception as exc:
            log.warning("Error while resetting device on disconnect: %s", exc)
        self.ccs = None
        self.wavelength = None
        log.info("Disconnected")
        self._set_ui_state(connected=False, running=False)

    def apply_integration_time(self, value: float):
        if self.ccs is None:
            return
        try:
            self.ccs.set_integration_time(value)
        except Exception as exc:
            self._show_error("Failed to Set Integration Time", str(exc))
            return

        # If a scan is running, re-pace the worker so it keeps polling
        # (and therefore logging) at the new integration time.
        if self.acq_worker is not None:
            self.acq_worker.poll_interval_ms = max(
                int(value * 1000), MIN_POLL_INTERVAL_MS
            )

    # ------------------------------------------------------------------
    # CSV logger
    # ------------------------------------------------------------------
    def choose_log_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Logging Folder", self.log_folder or ""
        )
        if not folder:
            return
        self.log_folder = folder
        # Elide long paths so the label doesn't stretch the layout.
        shown = folder if len(folder) <= 48 else "…" + folder[-45:]
        self.log_folder_label.setText(shown)
        self.log_folder_label.setToolTip(folder)

    def toggle_logging(self):
        if self.logging_active:
            self.stop_logging()
        else:
            self.start_logging()

    def start_logging(self):
        if self.ccs is None or self.wavelength is None:
            self._show_error("Cannot Start Logging",
                              "Connect to the spectrometer first, so a "
                              "wavelength axis is available for each file.")
            return

        if self.log_folder is None:
            self.choose_log_folder()
            if self.log_folder is None:
                return

        self.log_row_count = 0
        self.logging_active = True
        self.log_toggle_btn.setText("■  Stop Logging")
        self._restyle_button(self.log_toggle_btn, "stopBtn")
        self.log_status_label.setText("● LOGGING · 0 files")
        log.info("Logging started -> %s", self.log_folder)
        self._set_ui_state(connected=self.ccs is not None,
                            running=self.acq_worker is not None)

    def stop_logging(self):
        if self.logging_active:
            log.info("Logging stopped: %d file(s) written to %s",
                      self.log_row_count, self.log_folder)

        self.logging_active = False
        self.log_toggle_btn.setText("●  Start Logging")
        self._restyle_button(self.log_toggle_btn, "primaryBtn")
        self._set_ui_state(connected=self.ccs is not None,
                            running=self.acq_worker is not None)

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------
    def start_acquisition(self):
        if self.ccs is None:
            return

        try:
            self.ccs.set_integration_time(self.int_time.value())
            self.ccs.start_continuous_scan()
        except Exception as exc:
            self._show_error("Failed to Start Acquisition", str(exc))
            return

        poll_ms = max(int(self.int_time.value() * 1000), MIN_POLL_INTERVAL_MS)
        self.acq_thread = QThread(self)
        self.acq_worker = AcquisitionWorker(self.ccs, poll_ms)
        self.acq_worker.moveToThread(self.acq_thread)

        self.acq_thread.started.connect(self.acq_worker.start)
        self.acq_worker.spectrum_ready.connect(self._on_spectrum)
        self.acq_worker.error.connect(self._on_acquisition_error)

        self.acq_thread.start()
        self._set_ui_state(connected=True, running=True)

    def stop_acquisition(self):
        if self.acq_worker is not None:
            self.acq_worker.stop()
        if self.acq_thread is not None:
            self.acq_thread.quit()
            self.acq_thread.wait(2000)
        self.acq_thread = None
        self.acq_worker = None

        if self.ccs is not None:
            try:
                self.ccs.reset()
            except Exception as exc:
                log.warning("Error while resetting device on stop: %s", exc)

        self._set_ui_state(connected=self.ccs is not None, running=False)

    def _on_spectrum(self, spectrum: np.ndarray):
        self.last_spectrum = spectrum
        self.curve.setData(self.wavelength, spectrum)

        peak_idx = int(np.argmax(spectrum))
        peak_wl = self.wavelength[peak_idx]
        peak_val = spectrum[peak_idx]
        self.peak_marker.setData([peak_wl], [peak_val])
        self.status_peak.setText(f"PEAK: {peak_wl:.2f} nm @ {peak_val:.1f}")

        self.save_btn.setEnabled(True)
        self.save_png_btn.setEnabled(True)

        self._log_spectrum(spectrum)

    def _log_spectrum(self, spectrum: np.ndarray):
        """Write one CSV file per scan into the logging folder.

        One integration time = one file, so an integration time of 1s
        run for 1 minute of logging produces 60 files. Each file uses
        the exact same two-column format as the manual "Save CSV"
        snapshot: a wavelength_nm/intensity header followed by one row
        per wavelength sample.
        """
        if not self.logging_active or self.log_folder is None:
            return

        # Millisecond-resolution timestamp + a running index, so files
        # stay uniquely named even when integration time is well under 1s.
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"spectrum_{self.log_row_count + 1:05d}_{stamp}.csv"
        filepath = os.path.join(self.log_folder, filename)

        try:
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["wavelength_nm", "intensity"])
                for wl, val in zip(self.wavelength, spectrum):
                    writer.writerow([wl, val])
        except OSError as exc:
            self._show_error("Logging Failed", str(exc))
            self.stop_logging()
            return

        self.log_row_count += 1
        self.log_status_label.setText(
            f"● LOGGING · {self.log_row_count} files · last: {filename}"
        )

    def _on_acquisition_error(self, message: str):
        self.stop_acquisition()
        self._show_error("Acquisition Error", message)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def save_csv(self):
        if self.last_spectrum is None:
            return

        default_name = datetime.now().strftime("spectrum_%Y%m%d_%H%M%S.csv")
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Spectrum", default_name, "CSV Files (*.csv)"
        )
        if not filename:
            return

        try:
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["wavelength_nm", "intensity"])
                for wl, val in zip(self.wavelength, self.last_spectrum):
                    writer.writerow([wl, val])
        except OSError as exc:
            self._show_error("Save Failed", str(exc))
            return

        log.info("Saved CSV: %s", filename)
        self.statusBar().showMessage(f"Saved {filename}", 4000)

    def save_png(self):
        if self.last_spectrum is None:
            return

        default_name = datetime.now().strftime("spectrum_%Y%m%d_%H%M%S.png")
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Plot Image", default_name, "PNG Files (*.png)"
        )
        if not filename:
            return

        try:
            exporter = pg.exporters.ImageExporter(self.plot.plotItem)
            exporter.export(filename)
        except Exception as exc:
            self._show_error("Save Failed", str(exc))
            return

        log.info("Saved PNG: %s", filename)
        self.statusBar().showMessage(f"Saved {filename}", 4000)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def closeEvent(self, event):
        self.stop_acquisition()
        if self.logging_active:
            self.stop_logging()
        if self.ccs is not None:
            try:
                self.ccs.reset()
            except Exception as exc:
                log.warning("Error while resetting device on close: %s", exc)
        event.accept()


def main():
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    app.setFont(QFont("Segoe UI", 9))

    window = CCSWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()