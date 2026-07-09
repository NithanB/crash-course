import numpy as np
import matplotlib.pyplot as plt

np.random.seed(88) # fix the specific random seed for reproducibility

# main parameters
N = 1024 # number of samples
band_limit = 50 # band limit for the signal


#Generate random Fourier coefficients
coefficients = np.zeros(N, dtype=complex) # initialize the coefficients array with zeros
#the function is real,. ensure the symmetry of the coefficients

coefficients[1:band_limit] = np.random.randn(band_limit - 1) + 1j * np.random.randn(band_limit - 1) 
# generate random complex coefficients for positive frequencies


# ensure the symmetry for negative frequencies which is conjugate frequency

coefficients[-(band_limit-1):] = coefficients[1:band_limit][::-1].conj() # set negative frequencies to be the conjugate of positive frequencies

# plot graphs

# inverse fft to generate the band-limited time-domain function
time_domain_function = np.fft.ifft(coefficients).real


plt.figure(figsize = (10, 6))
plt.plot(time_domain_function, label="Band Limited Function", color='green')
plt.title("Band Limited Function from Random Fourier Coefficients"  )
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.legend()
plt.show()
