import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Parameters
# -------------------------------
wavelength = 633e-9       # wavelength [m] (red HeNe laser)
k = 2 * np.pi / wavelength
z = 0.1                   # propagation distance [m]
N = 512                   # grid size
dx = 10e-6                # pixel pitch [m]
L = N * dx                # physical size [m]
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

# -------------------------------
# Step 1: Define Object
# -------------------------------
def create_object():
    """Create a circular aperture object."""
    R = 200e-6   # radius of circle [m]
    obj = np.zeros((N, N))
    obj[np.sqrt(X**2 + Y**2) < R] = 1.0
    return obj

object_amp = create_object()

# -------------------------------
# Step 2: Forward Fresnel Propagation (Hologram)
# -------------------------------
def fresnel_propagation(U0, z, wavelength, dx):
    """Fresnel propagation using Fourier method."""
    N = U0.shape[0]
    k = 2 * np.pi / wavelength
    fx = np.fft.fftfreq(N, d=dx)
    fy = np.fft.fftfreq(N, d=dx)
    FX, FY = np.meshgrid(fx, fy)
    H = np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))
    U1 = np.fft.ifft2(np.fft.fft2(U0) * H)
    return U1

# object field (amplitude only for simplicity)
U0 = object_amp

# Propagate to distance z to create hologram
hologram_field = fresnel_propagation(U0, z, wavelength, dx)
hologram_intensity = np.abs(hologram_field)**2

# -------------------------------
# Step 3: Reconstruction (Back-propagation)
# -------------------------------
reconstructed_field = fresnel_propagation(hologram_field, -z, wavelength, dx)
reconstructed_amp = np.abs(reconstructed_field)

# -------------------------------
# Step 4: Display Results
# -------------------------------
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(object_amp, cmap='gray')
plt.title("Original Object")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(hologram_intensity, cmap='gray')
plt.title("Simulated Hologram")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(reconstructed_amp, cmap='gray')
plt.title("Reconstruction")
plt.axis("off")

plt.tight_layout()
plt.savefig("figures")
plt.show()
