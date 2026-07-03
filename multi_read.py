from thorlabs_ccs import *
import matplotlib.pyplot as plt
import array

spectro = list_spectrometers()
ccs200 = TLCCS(
    device_info = spectro[0]
)

ccs200.set_integration_time(0.4)

fig, ax = plt.subplots()
wavelength = ccs200.get_wavelength()
line, = ax.plot(wavelength, array.array('f', [0]*len(wavelength)))
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Normalized Intensity")
ax.set_ylim(-0.01, 1.1)
plt.ion()
plt.show()

ccs200.start_continuous_scan()

try:
    while True:
        
        spectrum = ccs200.get_scan_data_factory()
        line.set_ydata(spectrum)
        ax.set_ylim(-0.01, 1.1*max(spectrum))
        fig.canvas.draw()
        fig.canvas.flush_events()

except KeyboardInterrupt:
    print("Stopping acquisition...")
    ccs200.reset()  
    plt.ioff()
    plt.close(fig)
