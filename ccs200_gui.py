import sys
import csv
from datetime import datetime

import numpy as np
import pyqtgraph as pg

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QDoubleSpinBox,
    QFileDialog,
)

from thorlabs_ccs import *


class CCSWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.ccs = None
        self.wavelength = None
        self.last_spectrum = None

        self.setWindowTitle("Thorlabs CCS200 Viewer")
        self.resize(1200, 800)

        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)

        controls = QHBoxLayout()
        layout.addLayout(controls)

        self.connect_btn = QPushButton("Connect")
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.save_btn = QPushButton("Save CSV")

        controls.addWidget(self.connect_btn)
        controls.addWidget(self.start_btn)
        controls.addWidget(self.stop_btn)
        controls.addWidget(self.save_btn)

        controls.addWidget(QLabel("Integration Time (s)"))

        self.int_time = QDoubleSpinBox()
        self.int_time.setDecimals(4)
        self.int_time.setRange(0.00001, 60)
        self.int_time.setValue(0.4)

        controls.addWidget(self.int_time)

        self.plot = pg.PlotWidget()

        self.plot.showGrid(x=True, y=True)

        self.plot.setLabel(
            "bottom",
            "Wavelength",
            units="nm"
        )

        self.plot.setLabel(
            "left",
            "Intensity"
        )

        layout.addWidget(self.plot)

        self.curve = self.plot.plot(
            pen='y'
        )

        self.timer = QTimer()

        self.connect_btn.clicked.connect(
            self.connect_spectrometer
        )

        self.start_btn.clicked.connect(
            self.start_acquisition
        )

        self.stop_btn.clicked.connect(
            self.stop_acquisition
        )

        self.save_btn.clicked.connect(
            self.save_csv
        )

        self.timer.timeout.connect(
            self.update_spectrum
        )

    def connect_spectrometer(self):

        devices = list_spectrometers()

        if not devices:
            print("No spectrometer found")
            return

        self.ccs = TLCCS(
            device_info=devices[0]
        )

        self.wavelength = np.array(
            self.ccs.get_wavelength()
        )

        print("Connected")

    def start_acquisition(self):

        if self.ccs is None:
            return

        self.ccs.set_integration_time(
            self.int_time.value()
        )

        self.ccs.start_continuous_scan()

        self.timer.start(50)

    def stop_acquisition(self):

        self.timer.stop()

        if self.ccs:
            self.ccs.reset()

    def update_spectrum(self):

        try:

            spectrum = np.array(
                self.ccs.get_scan_data_factory()
            )

            self.last_spectrum = spectrum

            self.curve.setData(
                self.wavelength,
                spectrum
            )

        except Exception as e:

            print(e)

    def save_csv(self):

        if self.last_spectrum is None:
            return

        default_name = datetime.now().strftime(
            "spectrum_%Y%m%d_%H%M%S.csv"
        )

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Spectrum",
            default_name,
            "CSV Files (*.csv)"
        )

        if not filename:
            return

        with open(
            filename,
            "w",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                ["wavelength_nm", "intensity"]
            )

            for wl, val in zip(
                self.wavelength,
                self.last_spectrum
            ):
                writer.writerow([wl, val])

        print("Saved:", filename)


if __name__ == "__main__":

    app = QApplication(sys.argv)

    pg.setConfigOptions(
        antialias=True
    )

    window = CCSWindow()

    window.show()

    sys.exit(app.exec_())
