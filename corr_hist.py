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


def corr_plot(foxsi, norm = LogNorm(), cmapp = 'viridis', foxsi_label = 'test', binss = 60):
	
	fig, ax = plt.subplots(2, 3, figsize = (12, 8))
	ax[0][0].hist2d(foxsi.flatten(), aia_171_small.flatten(), bins = binss, norm = norm, cmap = cmapp)
	ax[0][0].set_xlabel('I (FOXSI 3-5keV)')
	ax[0][0].set_ylabel('I (AIA 171 $\mathrm{\AA}$)')

	ax[0][1].hist2d(foxsi.flatten(), aia_193_small.flatten(), bins = binss, norm = norm, cmap = cmapp)
	ax[0][1].set_xlabel('I (FOXSI 3-5keV)')
	ax[0][1].set_ylabel('I (AIA 193 $\mathrm{\AA}$)')


	ax[0][2].hist2d(foxsi.flatten(), aia_211_small.flatten(), bins = binss, norm = norm, cmap = cmapp)

	ax[0][2].set_xlabel('I (FOXSI 3-5keV)')
	ax[0][2].set_ylabel('I (AIA 211 $\mathrm{\AA}$)')


	ax[1][0].hist2d(foxsi.flatten(), aia_335_small.flatten(), bins = binss, norm = norm, cmap = cmapp)

	ax[1][0].set_xlabel('I (FOXSI 3-5keV)')
	ax[1][0].set_ylabel('I (AIA 335 $\mathrm{\AA}$)')


	ax[1][1].hist2d(foxsi.flatten(), aia_94_small.flatten(), bins = binss, norm = norm, cmap = cmapp)
	ax[1][1].set_xlabel('I (FOXSI 3-5keV)')
	ax[1][1].set_ylabel('I (AIA 94 $\mathrm{\AA}$)')

	ax[1][2].hist2d(foxsi.flatten(), aia_131_small.flatten(), bins = binss, norm = norm, cmap = cmapp)
	ax[1][2].set_xlabel('I ('+foxsi_label+')')
	ax[1][2].set_ylabel('I (AIA 131 $\mathrm{\AA}$)')

	plt.tight_layout()


def aia_corr_conf():
	#fig, ax = plt.subplots(figsize= (10,10))
	test_aia = [aia171, aia193, aia211, aia335, aia94, aia131]
	testn = ['171 $\mathrm{\AA}$', '193 $\mathrm{\AA}$', '211 $\mathrm{\AA}$', '335 $\mathrm{\AA}$', '94 $\mathrm{\AA}$', '131 $\mathrm{\AA}$']
	xx = np.zeros((6,6))
	for i in range(len(test_aia)):
		for j in range(len(test_aia)):
	    	  
			print(testn[i], testn[j],  np.corrcoef(test_aia[i].flatten(), test_aia[j].flatten())[0][1])
			xx[i][j] = np.corrcoef(test_aia[i].flatten(), test_aia[j].flatten())[0][1]


	cmap = 'viridis_r'
	plt.imshow(xx, interpolation='nearest', cmap=cmap)
	plt.title('Correlation Coefficients AIA channels')
	cbar = plt.colorbar()
	cbar.set_label('Correlation Coefficient')
	tick_marks = np.arange(len(testn))
	plt.xticks(tick_marks, testn, rotation=45)
	plt.yticks(tick_marks, testn)

	for i, j in itertools.product(range(xx.shape[0]), range(xx.shape[1])):
	    plt.text(j, i, format(cm[i, j], '.2f'),
	             horizontalalignment="center",
	             color="k")


def aia_foxsi_corr_conf():
	fig, ax = plt.subplots(figsize = (10, 7))
	test_aia = [aia_171_small, aia_193_small, aia_211_small, aia_335_small, aia_94_small, aia_131_small]
	test_foxsi = [foxsi_35, foxsi_510, foxsi_1025]
	testn = ['171 $\mathrm{\AA}$', '193 $\mathrm{\AA}$', '211 $\mathrm{\AA}$', '335 $\mathrm{\AA}$', '94 $\mathrm{\AA}$', '131 $\mathrm{\AA}$']
	testf = ['3-5 keV', '5-10 keV', '10-25 keV']
	xx = np.zeros((3,6))
	for i in range(len(test_foxsi)):
		for j in range(len(test_aia)):
	    	  
			print(testf[i], testn[j],  np.corrcoef(test_foxsi[i].flatten(), test_aia[j].flatten())[0][1])
			xx[i][j] = np.corrcoef(test_foxsi[i].flatten(), test_aia[j].flatten())[0][1]


	cmap = 'viridis_r'
	im = plt.imshow(xx, interpolation='nearest', cmap=cmap)
	plt.title('Correlation Coefficients FOXSI and AIA channels')
	cbar = plt.colorbar()
	cbar.set_label('Correlation Coefficient')
	xtick_marks = np.arange(len(testn))
	ytick_marks = np.arange(len(testf))
	plt.xticks(xtick_marks, testn, rotation=45)
	plt.yticks(ytick_marks, testf)

	for i, j in itertools.product(range(xx.shape[0]), range(xx.shape[1])):
	    plt.text(j, i, format(xx[i, j], '.2f'),
	             horizontalalignment="center",
	             color="k")


def temp_foxsi_corr_conf():
	fig, ax = plt.subplots(figsize = (10, 7))
	
	test_foxsi = [foxsi_35, foxsi_510, foxsi_1025]
	tm = np.round(aia['tm']/1e6, 2)
	tmm = [str(x) + ' MK' for x in tm]
	testf = ['3-5 keV', '5-10 keV', '10-25 keV']
	xx = np.zeros((3,len(aia['dem'])))
	for i in range(len(test_foxsi)):
		for j in range(len(aia['dem'])):
	    	  
			print(testf[i], tmm[j],  np.corrcoef(test_foxsi[i].flatten(), imresize(aia['dem'][j], (67, 67)).flatten())[0][1])
			xx[i][j] = np.corrcoef(test_foxsi[i].flatten(), imresize(aia['dem'][j], (67, 67)).flatten())[0][1]


	cmap = 'viridis_r'
	im = plt.imshow(xx, interpolation='nearest', cmap=cmap)
	plt.title('Correlation Coefficients FOXSI and AIA channels')
	xtick_marks = np.arange(len(tmm))
	ytick_marks = np.arange(len(testf))
	plt.xticks(xtick_marks, tmm, rotation=45)
	plt.yticks(ytick_marks, testf)



	for i, j in itertools.product(range(xx.shape[0]), range(xx.shape[1])):
	    plt.text(j, i, format(xx[i, j], '.2f'),
	             horizontalalignment="center",
	             color="k")

	divider = make_axes_locatable(ax)
	cax = divider.append_axes("bottom", size="5%", pad=0.05)
	cbar = plt.colorbar(orientation = 'horizontal', ax = cax)
	cbar.set_label('Correlation Coefficient')

def corr_logged():
	test_aia = [aia_171_small, aia_193_small, aia_211_small, aia_335_small, aia_94_small, aia_131_small]
	test_foxsi = [foxsi_35, foxsi_510, foxsi_1025]
	testn = ['171 $\mathrm{\AA}$', '193 $\mathrm{\AA}$', '211 $\mathrm{\AA}$', '335 $\mathrm{\AA}$', '94 $\mathrm{\AA}$', '131 $\mathrm{\AA}$']
	testf = ['3-5 keV', '5-10 keV', '10-25 keV']

def calc_shift(image1, image2):
	shift, error, diffphase = register_translation(image1, image2)
	print(shift)



	unrolled = np.roll(np.roll(image2, int(shift[1])), int(shift[0]), axis = 0)

	shift2, error2, diffphase2 = register_translation(image1, unrolled)
	print(shift2)
	return unrolled




t1 = calc_shift(aia_131_small, foxsi_35)
t2 = calc_shift(aia_335_small, foxsi_35)
t3 = calc_shift(aia_171_small, foxsi_35)
t4 = calc_shift(aia_94_small, foxsi_35)
t5 = calc_shift(aia_193_small, foxsi_35)
t6 = calc_shift(aia_211_small, foxsi_35)

