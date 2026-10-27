import numpy as np

# 'physical' things
eV = 1.60217662e-19  # J or charge of electron in C
GeV = eV * 1e9  # J
J = 1 / GeV  # GeV
hbarGeV = 6.582119569e-16 * 1e-9  # GeV s
c = 3e8  # m/s

kg = c**2 / GeV  # GeV
g = kg / 1000  # GeV
m = 1 / (hbarGeV * c)  # GeV^-1
s = 1 / hbarGeV  # GeV^-1
G = 6.674e-11 * m**3 / kg / s**2 # GeV^-2

# convert kg to GeV by multiplying kg mass by 'kg'
mp = 1.67262192369e-27 * kg  # GeV
mn = 1.674927471e-27 * kg  # GeV

eps0 = 8.854187817e-12 * kg * m  # * C^2
mu0 = 4e-7 * np.pi * s**2 / kg / m**3  # / C^2

kB = 1.38064852e-23 * J  # GeV / K

# barns typically used for cross sections, 1 barn = 10^-24 cm^2
barn = 1e-28 * m**2  # GeV^-2

# cosmo/astro helpful things
Mpl = 1 / np.sqrt(G)  # GeV
mpl = 1 / np.sqrt(8*np.pi * G)  # GeV
# print(Mpl / g)  # e.g. convert to grams
pc = 3.086e16 * m  # Gev^-1
hH = 0.7
H0 = hH * 100 * 1000 / (pc * 1e6) / s  # GeV
Gyr = 3600 * 24 * 365e9 * s # GeV^-1

rhocrit0 = 1.05375e-5 / (m / 100)**3  # GeV^4