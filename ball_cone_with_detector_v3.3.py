from __future__ import division
import numpy as np
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as plt

# Constants
c = 3.e10  # [cm / s]
h = 6.63e-27 #planck's constant [cgs]
k = 1.38e-16 #Boltzman's constant [cgs]
H_0 = 65 #Hubble constant

#Parameters
beta = 0.9999  # v/c
R1 = 1e13  # cm
R2 = 2e13  # cm
T = 1e7 #Comoving Temperature of fireball [K]

# Initialize a number of blobs
blobs = 1000

#Total energy of fireball
E = 10**53 #[erg]

#Efficiency parameter -> what fraction of energy is radiated (10-20%)
epsilon = 0.1

#Detector properties
#Define detectable frequency range
nu_D_min = 1e16 #[Hz]
nu_D_max = 1e22 #[Hz]

# Define jet opening angle
theta_j = (np.pi / 180) * 10

#Energy per blob
E_i = E / blobs

# Initialize GRB simulation time
t = 10**np.arange(-2, 3, 0.00001) #[s]

# Define observer unit vector
obs_theta = 0
obs_phi = 0

# Initialize theta of blobs
x = np.random.rand(blobs)
theta = np.arccos((1 - np.cos(theta_j)) * x + np.cos(theta_j))

# Initialize Phi of blobs
phi = np.random.rand(blobs) * 2 * np.pi

##########################################################################################################################

# Define tus as a function of theta and r
def tus(theta, r):
    return (r / c) * (1 - beta * np.cos(theta))

# Define Gamma_blob as a function of beta
def Gamma(beta):
    return 1 / np.sqrt(1 - beta**2)


# Define the doppler factor
def Delta(beta, theta):
    return 1 / (Gamma(beta) * (1 - beta * np.cos(theta)))


# Define the spectral intensity function as a funtion of nu in the blob frame (Planck's Law)
def l(nu):
	return ((2*h / c**2) * (nu**3) / (np.exp(h*nu/(k * T)) - 1)) # Watt/(Sr m**2 Hz) ---> Some of these numbers are too big


#Constant of normalization
#C = (epsilon * E_i * 60  * (R2 - R1) * h**4 ) / (Gamma(beta)**2 * beta * c * np.pi**3 * k**4 * T**4)
C = (15 * c**3 * h**3 * beta * epsilon * E_i) / (8 * np.pi**5 * (R2 - R1) * k**4 * T**4)

##########################################################################################################################
cosmo = FlatLambdaCDM(H0=H_0, Om0=0.3)

deltaOmega = 4*np.pi #Solid angle covered by sky survey

def Ngrb(z):
	return 1e-3 * (c/H_0) * deltaOmega * 0.2 * H_0 * np.array(cosmo.luminosity_distance(z))**2 * np.exp(3.05*z - 0.4) / \
	((np.exp(2.93*z) + 15) * ((1+z)**3 * (0.3*(1+z)**3 + 0.7)**0.5))

##########################################################################################################################

Zrand = 10 * np.random.rand(100000)

Nrand = np.amax(Ngrb(Zrand)) * np.random.rand(Zrand.size)

jj = np.where(Ngrb(Zrand) >= Nrand)

zlist = Zrand[jj]

##########################################################################################################################

def flux(z, L):
	f = (L * u.Watt / (4*np.pi*cosmo.luminosity_distance(z)**2))
	return f.to(u.erg / (u.s * u.cm**2)) # ergs/(s cm**2 sr)

##########################################################################################################################

def Lum():
	# Initialize integrated Luminosity as a zero array
	Lum = np.zeros(t.shape)

	for i in xrange(0, blobs):
		blob_vec = np.array([np.cos(phi[i]) * np.sin(theta[i]), np.sin(phi[i]) * np.sin(theta[i]), np.cos(theta[i])])
		obs_vec = np.array([np.cos(obs_phi) * np.sin(obs_theta), np.sin(obs_phi) * np.sin(obs_theta), np.cos(obs_theta)])

		tot_obs_angle = np.arccos(np.dot(blob_vec, obs_vec))

		ton = tus(tot_obs_angle, R1)
		toff = tus(tot_obs_angle, R2)

		nu = np.logspace(np.log10(nu_D_min), np.log10(nu_D_max), 100000)
		nu_prime = (1/Delta(beta, tot_obs_angle))*nu

		Lblob = 0
		Lblob = C * np.trapz(Delta(beta, tot_obs_angle)**3 * l(nu_prime), x=nu_prime)

		j = np.where((t >= ton) & (t <= toff))
		Lum[j] += Lblob
	return Lum

##########################################################################################################################

GRBs = 2500

years = 1

Flux = np.array([])

t_grb = np.array([])

for i in xrange(0,GRBs):
	obs_theta = np.random.rand() * (np.pi/2)
	obs_phi = 0
	if obs_theta > theta_j + (np.pi/180)*2:
		F = flux(zlist[i],Lum())
		t_grb = np.append(t_grb, np.random.randint(years * 365))
		Flux = np.append(Flux, np.sum(F) / 86400)
		print 'GRB# = %d, Obs angle (deg) = %f, z = %f' % (i, obs_theta * (180/np.pi), zlist[i])
		np.savetxt('Run_2500_1yr_100_200keV_b.dat', (t_grb, Flux))
	else:
		print "Throwing out near on-axis observation of GRB"

# plt.figure()
# plt.loglog(t, Lum())
# plt.show()