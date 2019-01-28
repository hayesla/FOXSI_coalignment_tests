import numpy as np 
import matplotlib.pyplot as plt 
from scipy.io import readsav
from sunpy import map
import datetime 
from matplotlib.colors import LogNorm, Normalize
from scipy.misc import imresize
from skimage.transform import resize
import copy
from scipy import signal
import seaborn as sns
sns.set_context('paper')
sns.set_palette('dark')
from mpl_toolkits.axes_grid1 import make_axes_locatable


from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft
from scipy.ndimage import fourier_shift

import itertools

def norm(x):
	return (x - np.min(x))/(np.max(x) - np.min(x))
	#return (x - np.mean(x))



foxsi_file = 'joel_ar_foxsi_images.sav'
sim_aia_file = 'aia_images_for_steven.sav'

foxsi = readsav(foxsi_file)
foxsi_35 = foxsi['outmap_3_5']['data'][0]
foxsi_510 = foxsi['outmap_5_10']['data'][0]
foxsi_1025 = foxsi['outmap_10_25']['data'][0]

aia = readsav(sim_aia_file)
aia171 = aia['syn171']
aia131 = aia['syn131']
aia211 = aia['syn211']
aia94 = aia['syn94']
aia335 = aia['syn335']
aia193 = aia['syn193']

foxsi_35_resize = resize(foxsi_35, (418, 418))
foxsi_510_resize = resize(foxsi_510, (418, 418))
foxsi_1025_resize = resize(foxsi_1025, (418, 418))

aia_171_small = imresize(aia171, (67,67))
aia_131_small = imresize(aia131, (67,67))
aia_211_small = imresize(aia211, (67,67))
aia_94_small = imresize(aia94, (67,67))
aia_335_small = imresize(aia335, (67,67))
aia_193_small = imresize(aia193, (67,67))

xa = [aia['x'][0], aia['x'][-1], aia['y'][0], aia['y'][-1]]

def calc_shift(image1, image2):
	shift, error, diffphase = register_translation(image1, image2)
	print(shift)



	unrolled = np.roll(np.roll(image2, int(shift[1])), int(shift[0]), axis = 0)

	shift2, error2, diffphase2 = register_translation(image1, unrolled)
	print(np.corrcoef(image1.flatten(), image2.flatten())[0][1], shift2)
	return shift, unrolled

shift1_35, f1_35 = calc_shift(aia_131_small, foxsi_35)
shift2_35, f2_35 = calc_shift(aia_171_small, foxsi_35)
shift3_35, f3_35 = calc_shift(aia_193_small, foxsi_35)
shift4_35, f4_35 = calc_shift(aia_211_small, foxsi_35)
shift5_35, f5_35 = calc_shift(aia_335_small, foxsi_35)
shift6_35, f6_35 = calc_shift(aia_94_small, foxsi_35)


shift1_510, f1_510 = calc_shift(aia_131_small, foxsi_510)
shift2_510, f2_510 = calc_shift(aia_171_small, foxsi_510)
shift3_510, f3_510 = calc_shift(aia_193_small, foxsi_510)
shift4_510, f4_510 = calc_shift(aia_211_small, foxsi_510)
shift5_510, f5_510 = calc_shift(aia_335_small, foxsi_510)
shift6_510, f6_510 = calc_shift(aia_94_small, foxsi_510)


shift1_1025, f1_1025 = calc_shift(aia_131_small, foxsi_1025)
shift2_1025, f2_1025 = calc_shift(aia_171_small, foxsi_1025)
shift3_1025, f3_1025 = calc_shift(aia_193_small, foxsi_1025)
shift4_1025, f4_1025 = calc_shift(aia_211_small, foxsi_1025)
shift5_1025, f5_1025 = calc_shift(aia_335_small, foxsi_1025)
shift6_1025, f6_1025 = calc_shift(aia_94_small, foxsi_1025)


def test():
	h = plt.hist2d(norm(foxsi_35).flatten(), norm(aia_335_small).flatten(), bins = 60)
	z = h[0]
	y, x = np.indices(z.shape)

	x = x.ravel()
	y = y.ravel()
	z = z.ravel()

	z2 = np.polyfit(x, y, w = z**0.5, deg = 1, cov = True)
	p = np.poly1d(z2[0])

	x_plot = np.linspace(x.min(), x.max(), 100)
	y_plot = p(x_plot)

	plt.imshow(h[0], origin = 'lower', norm = LogNorm())
	plt.plot(x_plot, y_plot, color = 'r')




def plot_corr():
	corry = signal.correlate2d(aia_335_small, foxsi_35, mode = 'same')
	maxy = np.unravel_index(np.argmax(corry), corry.shape)
	fig, ax = plt.subplots(1, 3, figsize = (15,5))
	ax[0].imshow(aia335, extent = xa, cmap ='sdoaia335', origin = 'lower', vmax = np.mean(aia335)+6*np.std(aia335))
	ax[0].set_xlabel('Solar X (arcsec)')
	ax[0].set_ylabel('Solar Y (arcsec)')
	ax[0].set_title(r'AIA 335 $\mathrm{\AA}$')
	ax[1].imshow(foxsi_35, extent = xa, cmap = 'viridis',  origin = 'lower', vmax = np.mean(foxsi_35)+6*np.std(foxsi_35))
	ax[1].set_ylabel('Solar Y (arcsec)')
	ax[1].set_xlabel('Solar X (arcsec)')
	ax[1].set_title('FOXSI 3-5 keV')
	ax[2].imshow(corry, cmap = 'magma', origin = 'lower')
	ax[2].set_title('Correlation Map')
	ax[2].set_xlabel('Shift X pixel')
	ax[2].set_ylabel('Shift Y pixel')
	ax[2].axvline(34.5)
	ax[2].axhline(34.5)
	ax[2].plot(maxy[1], maxy[0], color = 'b', ls = ' ', marker = '.')




def plot_tog():
	#levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])*np.max(foxsi_35)
	#levels_35 = np.linspace(0.05, 1, 20)*np.max(foxsi_1025)
	levels_35 = np.linspace(0.2, 1, 20)*np.max(foxsi_35)
	fig, ax = plt.subplots(1, 3, figsize = (15, 5))

	ax[0].imshow(aia335, extent = xa, cmap ='sdoaia335', origin = 'lower', vmax = np.mean(aia335)+6*np.std(aia335))
	ax[0].set_xlabel('Solar X (arcsec)')
	ax[0].set_ylabel('Solar Y (arcsec)')
	ax[0].set_title(r'AIA 335 $\mathrm{\AA}$')
	ax[1].imshow(foxsi_35, extent = xa, cmap = 'viridis',  origin = 'lower', vmax = np.mean(foxsi_35)+6*np.std(foxsi_35), norm = LogNorm())
	ax[1].set_ylabel('Solar Y (arcsec)')
	ax[1].set_xlabel('Solar X (arcsec)')
	ax[1].set_title('FOXSI 3-5 keV')

	ax[2].imshow(aia335, extent = xa, cmap ='sdoaia335', origin = 'lower', vmax = np.mean(aia335)+6*np.std(aia335))
	ax[2].set_ylabel('Solar Y (arcsec)')
	ax[2].set_xlabel('Solar X (arcsec)')
	ax[2].set_title('Plotted Together')
	ax[2].contour(foxsi_35, extent = xa, levels = levels_35, colors = 'r')


def plot_tog():
	fig, ax = plt.subplots(1, 3, figsize = (15, 5))
	ax[0].imshow(aia['syn335'], cmap= 'sdoaia335', vmax = aia['syn335'].mean() + 3*aia['syn335'].std(), extent = [xa[0], xa[1], ya[0], ya[-1]], origin = 'lower')
	ax[0].contour(foxsi_map_35.data, levels = levels_35, extent = [xa[0], xa[1], ya[0], ya[-1]], colors = 'red')
	ax[0].text(750, 330, 'FOXSI 3-5keV', color = 'r')

	ax[1].imshow(aia['syn94'], cmap= 'sdoaia94', vmax = aia['syn94'].mean() + 3*aia['syn94'].std(), extent = [xa[0], xa[1], ya[0], ya[-1]], origin = 'lower')
	ax[1].contour(foxsi_map_510.data, levels = levels_510, extent = [xa[0], xa[1], ya[0], ya[-1]], colors = 'blue')
	ax[1].text(750, 330, 'FOXSI 5-10keV', color = 'b')

	ax[2].imshow(aia['syn94'], cmap= 'sdoaia94', vmax = aia['syn94'].mean() + 3*aia['syn94'].std(), extent = [xa[0], xa[1], ya[0], ya[-1]], origin = 'lower')
	ax[2].contour(foxsi_map_1025.data, levels = levels_1025, extent = [xa[0], xa[1], ya[0], ya[-1]], colors = 'green')
	ax[2].text(750, 330, 'FOXSI 10-25keV', color = 'g')



def corr_plot(foxsi, normm = LogNorm(), cmapp = 'viridis', foxsi_label = 'test', binss = 60, range = [[0,1],[0,1]]):
	
	fig, ax = plt.subplots(2, 3, figsize = (14, 8))
	h = ax[0][0].hist2d(norm(foxsi).flatten(), norm(aia_171_small).flatten(), bins = binss, norm = normm, cmap = cmapp, range = range)
	ax[0][0].set_xlabel('Normalized Intensity ('+foxsi_label+')')
	ax[0][0].set_ylabel('Normalized Intensity (AIA 171 $\mathrm{\AA}$)')
	#cbar = plt.colorbar(h[3] ,ax = ax[0][0])

	h1 = ax[0][1].hist2d(norm(foxsi).flatten(), norm(aia_193_small).flatten(), bins = binss, norm = normm, cmap = cmapp, range = range)
	ax[0][1].set_xlabel('Normalized Intensity ('+foxsi_label+')')
	ax[0][1].set_ylabel('Normalized Intensity (AIA 193 $\mathrm{\AA}$)')
	#cbar1 = plt.colorbar(h1[3] ,ax = ax[0][1])

	h2 = ax[0][2].hist2d(norm(foxsi).flatten(), norm(aia_211_small).flatten(), bins = binss, norm = normm, cmap = cmapp, range = range)

	ax[0][2].set_xlabel('Normalized Intensity ('+foxsi_label+')')
	ax[0][2].set_ylabel('Normalized Intensity (AIA 211 $\mathrm{\AA}$)')
	#cbar2 = plt.colorbar(h2[3] ,ax = ax[0][2])

	h3 = ax[1][0].hist2d(norm(foxsi).flatten(), norm(aia_335_small).flatten(), bins = binss, norm = normm, cmap = cmapp, range = range)

	ax[1][0].set_xlabel('Normalized Intensity ('+foxsi_label+')')
	ax[1][0].set_ylabel('Normalized Intensity (AIA 335 $\mathrm{\AA}$)')
	#cbar3 = plt.colorbar(h3[3] ,ax = ax[1][0])

	h4 = ax[1][1].hist2d(norm(foxsi).flatten(), norm(aia_94_small).flatten(), bins = binss, norm = normm, cmap = cmapp, range = range)
	ax[1][1].set_xlabel('Normalized Intensity ('+foxsi_label+')')
	ax[1][1].set_ylabel('Normalized Intensity (AIA 94 $\mathrm{\AA}$)')
	#cbar4 = plt.colorbar(h4[3] ,ax = ax[1][1])


	h5 = ax[1][2].hist2d(norm(foxsi).flatten(), norm(aia_131_small).flatten(), bins = binss, norm = normm, cmap = cmapp, range = range)
	ax[1][2].set_xlabel('Normalized Intensity ('+foxsi_label+')')
	ax[1][2].set_ylabel('Normalized Intensity (AIA 131 $\mathrm{\AA}$)')
	#cbar5 = plt.colorbar(h5[3] ,ax = ax[1][2])
	plt.tight_layout()
	fig.subplots_adjust(right=0.92)
	cbar_ax = fig.add_axes([0.95, 0.065, 0.01, 0.91])
	barr = fig.colorbar(h5[3], cax=cbar_ax)
	barr.set_label('Pixel Density')
	#plt.tight_layout()





