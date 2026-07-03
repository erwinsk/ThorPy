from thorlabs_ccs import *

spectro = list_spectrometers()

ccs200 = TLCCS(
    device_info = spectro[0]
)

ccs200.set_integration_time(0.1)
ccs200.start_single_scan()
spectrum = ccs200.get_scan_data_factory()