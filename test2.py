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


def norm(x):
	return (x - np.min(x))/(np.max(x) - np.min(x))



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

def plot_simple_corr():

	corr_aia_35 = []
	for i in [aia_171_small, aia_193_small, aia_211_small, aia_335_small, aia_94_small, aia_131_small]:
		corr_aia_35.append(np.corrcoef(foxsi_35.flat, i.flat)[0][1])

	corr_aia_510 = []
	for i in [aia_171_small, aia_193_small, aia_211_small, aia_335_small, aia_94_small, aia_131_small]:
		corr_aia_510.append(np.corrcoef(foxsi_510.flat, i.flat)[0][1])


	corr_aia_1025 = []
	for i in [aia_171_small, aia_193_small, aia_211_small, aia_335_small, aia_94_small, aia_131_small]:
		corr_aia_1025.append(np.corrcoef(foxsi_1025.flat, i.flat)[0][1])

	list_aia = ['AIA 171 $\mathrm{\AA}$', 'AIA 193 $\mathrm{\AA}$', 'AIA 211 $\mathrm{\AA}$', 'AIA 335 $\mathrm{\AA}$', 'AIA 94 $\mathrm{\AA}$', 'AIA 131 $\mathrm{\AA}$']
	x_arr = [1,2,3,4,5,6]

	plt.plot(x_arr, norm(corr_aia_35), color = 'r', label = 'FOXSI 3-5 keV '+ '('+str(round(np.max(corr_aia_35),2))+')', marker = '.')

	plt.plot(x_arr, norm(corr_aia_510), color = 'b', label = 'FOXSI 5-10 keV '+'('+str(round(np.max(corr_aia_510),2))+')' , marker = '.')

	plt.plot(x_arr, norm(corr_aia_1025), color = 'g', label = 'FOXSI 10-25 keV '+'('+str(round(np.max(corr_aia_1025),2))+')', marker = '.')
	plt.xticks(x_arr, list_aia)
	plt.legend()
	plt.ylabel('Normalized Correlation Coefficient')

def plot_simple_corr2():

	corr_aia_35 = []
	for i in [aia171, aia193, aia211, aia335, aia94, aia131]:
		corr_aia_35.append(np.corrcoef(foxsi_35_resize.flat, i.flat)[0][1])

	corr_aia_510 = []
	for i in [aia171, aia193, aia211, aia335, aia94, aia131]:
		corr_aia_510.append(np.corrcoef(foxsi_510_resize.flat, i.flat)[0][1])


	corr_aia_1025 = []
	for i in [aia171, aia193, aia211, aia335, aia94, aia131]:
		corr_aia_1025.append(np.corrcoef(foxsi_1025_resize.flat, i.flat)[0][1])

	list_aia = ['AIA 171 $\mathrm{\AA}$', 'AIA 193 $\mathrm{\AA}$', 'AIA 211 $\mathrm{\AA}$', 'AIA 335 $\mathrm{\AA}$', 'AIA 94 $\mathrm{\AA}$', 'AIA 131 $\mathrm{\AA}$']
	x_arr = [1,2,3,4,5,6]

	plt.plot(x_arr, norm(corr_aia_35), color = 'r', label = 'FOXSI 3-5 keV '+ '('+str(round(np.max(corr_aia_35),2))+')', marker = '.')

	plt.plot(x_arr, norm(corr_aia_510), color = 'b', label = 'FOXSI 5-10 keV '+'('+str(round(np.max(corr_aia_510),2))+')' , marker = '.')

	plt.plot(x_arr, norm(corr_aia_1025), color = 'g', label = 'FOXSI 10-25 keV '+'('+str(round(np.max(corr_aia_1025),2))+')', marker = '.')
	plt.xticks(x_arr, list_aia)
	plt.legend()
	plt.ylabel('Normalized Correlation Coefficient')	

dem = aia['dem']
dem2 = []
for i in range(len(dem)):
	a = imresize(dem[i], (67,67))
	dem2.append(a)

dem2 = np.array(dem2)

def corr_dem():
	cory_dem = []
	for i in range(len(dem)):
		cory_dem.append(np.corrcoef(dem[10].flat, dem[i].flat)[0][1])

	plt.plot(aia['tm']/1e7, cory_dem)

def foxsi_corr_dem():
	foxsi_corr_35 = []
	for i in range(len(dem2)):
		foxsi_corr_35.append(np.corrcoef(foxsi_35.flat, dem2[i].flat)[0][1])

	foxsi_corr_510 = []
	for i in range(len(dem2)):
		foxsi_corr_510.append(np.corrcoef(foxsi_510.flat, dem2[i].flat)[0][1])

	foxsi_corr_1025 = []
	for i in range(len(dem2)):
		foxsi_corr_1025.append(np.corrcoef(foxsi_1025.flat, dem2[i].flat)[0][1])

	best_temp_35 = aia['tm'][np.where(foxsi_corr_35 == np.max(foxsi_corr_35))[0][0]]
	best_temp_510 = aia['tm'][np.where(foxsi_corr_510 == np.max(foxsi_corr_510))[0][0]]
	best_temp_1025 = aia['tm'][np.where(foxsi_corr_1025 == np.max(foxsi_corr_1025))[0][0]]


	plt.plot(aia['tm']/1e6, foxsi_corr_35, label = 'FOXSI 3-5 keV', color = 'r')
	plt.plot(aia['tm']/1e6, foxsi_corr_510, label = 'FOXSI 5-10 keV', color = 'b')
	plt.plot(aia['tm']/1e6, foxsi_corr_1025, label = 'FOXSI 10-25 keV', color = 'g')
	plt.xlim(aia['tm'][0]/1e6, aia['tm'][-1]/1e6)
	plt.xlabel('Temperature (MK)')
	plt.ylabel('Correlation Coefficient')
	plt.legend()

def foxsi_corr_dem2():
	foxsi_corr_35 = []
	for i in range(len(dem2)):
		foxsi_corr_35.append(np.corrcoef(foxsi_35_resize.flat, dem[i].flat)[0][1])

	foxsi_corr_510 = []
	for i in range(len(dem2)):
		foxsi_corr_510.append(np.corrcoef(foxsi_510_resize.flat, dem[i].flat)[0][1])

	foxsi_corr_1025 = []
	for i in range(len(dem2)):
		foxsi_corr_1025.append(np.corrcoef(foxsi_1025_resize.flat, dem[i].flat)[0][1])

	best_temp_35 = aia['tm'][np.where(foxsi_corr_35 == np.max(foxsi_corr_35))[0][0]]
	best_temp_510 = aia['tm'][np.where(foxsi_corr_510 == np.max(foxsi_corr_510))[0][0]]
	best_temp_1025 = aia['tm'][np.where(foxsi_corr_1025 == np.max(foxsi_corr_1025))[0][0]]


	plt.plot(aia['tm']/1e6, foxsi_corr_35, label = 'FOXSI 3-5 keV', color = 'r')
	plt.plot(aia['tm']/1e6, foxsi_corr_510, label = 'FOXSI 5-10 keV', color = 'b')
	plt.plot(aia['tm']/1e6, foxsi_corr_1025, label = 'FOXSI 10-25 keV', color = 'g')
	plt.xlim(aia['tm'][0]/1e7, aia['tm'][-1]/1e7)
	plt.xlabel('Temperature (MK)')
	plt.ylabel('Correlation Coefficient')
	plt.legend()




