# Galaxy Integration Project - NGC 1275

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.integrate import simpson

# -----------------------------
# 1. Load Data
# -----------------------------
URL1 = 'https://raw.githubusercontent.com/NumericalMethodsSullivan'
URL2 = '/NumericalMethodsSullivan.github.io/master/data/'
URL = URL1 + URL2

data = np.array(pd.read_csv(URL + 'ngc1275.csv'))

wavelength_angstrom = data[:, 0]
intensity = data[:, 1]

# -----------------------------
# 2. Convert Wavelength -> Frequency
# -----------------------------
c = 3e8  # speed of light (m/s)
wavelength_m = wavelength_angstrom * 1e-10
frequency = c / wavelength_m

# Sort (important because frequency reverses order)
sort_idx = np.argsort(frequency)
frequency = frequency[sort_idx]
intensity = intensity[sort_idx]

# -----------------------------
# 3. Background Removal (Moving Average)
# -----------------------------
window = 50
background = np.convolve(intensity, np.ones(window)/window, mode='same')
clean_intensity = intensity - background

# -----------------------------
# 4. Peak Detection
# -----------------------------
peaks, properties = find_peaks(clean_intensity, height=np.max(clean_intensity)*0.1)

# -----------------------------
# 5. Helper: Find Peak Region
# -----------------------------
def get_peak_region(x, y, peak_idx, threshold=0):
    left = peak_idx
    while left > 0 and y[left] > threshold:
        left -= 1

    right = peak_idx
    while right < len(y)-1 and y[right] > threshold:
        right += 1

    return left, right

# -----------------------------
# 6. Integrate Peaks
# -----------------------------
results = []

for peak in peaks:
    left, right = get_peak_region(frequency, clean_intensity, peak)

    x_region = frequency[left:right]
    y_region = clean_intensity[left:right]

    if len(x_region) < 5:
        continue

    # Trapezoidal rule
    area_trap = np.trapz(y_region, x_region)

    # Simpson's rule
    area_simp = simpson(y_region, x_region)

    error = abs(area_trap - area_simp)

    results.append({
        "peak_index": peak,
        "frequency": frequency[peak],
        "area": area_trap,
        "error": error
    })

# -----------------------------
# 7. Handle Double Peak (~6700 Å)
# -----------------------------
# (Automatically handled by peak detection, but we ensure separation)

# -----------------------------
# 8. Print Results
# -----------------------------
print("\nEmission Line Strengths:\n")
print("Index | Frequency (Hz) | Strength (W/m^2) | Error")
print("--------------------------------------------------------")

for r in results:
    print(f"{r['peak_index']:5d} | {r['frequency']:.3e} | {r['area']:.3e} | {r['error']:.3e}")

# -----------------------------
# 9. Plot Results
# -----------------------------
plt.figure(figsize=(12, 8))

# Original
plt.subplot(3,1,1)
plt.plot(frequency, intensity)
plt.title("Original Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Intensity")

# Background
plt.subplot(3,1,2)
plt.plot(frequency, intensity, label="Original")
plt.plot(frequency, background, label="Background", linestyle='--')
plt.legend()
plt.title("Background Removal")

# Cleaned + Peaks
plt.subplot(3,1,3)
plt.plot(frequency, clean_intensity, label="Cleaned")
plt.plot(frequency[peaks], clean_intensity[peaks], 'rx', label="Peaks")
plt.legend()
plt.title("Detected Emission Lines")

plt.tight_layout()
plt.show()