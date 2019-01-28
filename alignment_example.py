import numpy as np 
import matplotlib.pyplot as plt 
from scipy.io import readsav
from sunpy import map
import datetime 
from matplotlib.colors import LogNorm, Normalize
from scipy.misc import imresize
from scipy import ndimage
smooth = ndimage.uniform_filter

foxsi_file = 'joel_ar_foxsi_images.sav'
sim_aia_file = 'aia_images_for_steven.sav'

foxsi = readsav(foxsi_file)
foxsi_35 = foxsi['outmap_3_5']
foxsi_510 = foxsi['outmap_5_10']
foxsi_1025 = foxsi['outmap_10_25']

xc = foxsi_35['xc']
yc = foxsi_35['yc']
dx = foxsi_35['dx']
dy = foxsi_35['dy']
time_obs = datetime.datetime.strptime(foxsi_35['time'].astype('str')[0], '%d-%b-%Y %H:%M:%S.%f')
xunits = foxsi_35['xunits'].astype('str')[0]
yunits = foxsi_35['yunits'].astype('str')[0]
roll_angle = foxsi_35['roll_angle'][0]

meta_foxsi = { 'crval1': xc[0],
		 'crval2': yc[0],
		 'cdelt1': dx[0],
		 'cdelt2': dy[0],
		 'date-obs': time_obs,
		 'cunit1': xunits, 
		 'cunit2':yunits,
		 'crota2': roll_angle,
		 'cmap' :plt.cm.viridis 

		}

foxsi_map_35 = map.Map(foxsi_35['data'][0], meta_foxsi)
foxsi_map_510 = map.Map(foxsi_510['data'][0], meta_foxsi)
foxsi_map_1025 = map.Map(foxsi_1025['data'][0], meta_foxsi)


xrange_foxsi = [xc[0] - 33*dx[0], xc[0]+33*dx[0]]
yrange_foxsi = [yc[0] - 33*dy[0], yc[0]+33*dy[0]]

def plot_foxsi(cmap = 'viridis', norm = Normalize()):
	fig, ax = plt.subplots(1,3, figsize = (15, 5))

	foxsi_map_35.plot(axes =ax[0], cmap = cmap, title = 'FOXSI 3-5 keV', norm  = norm)

	foxsi_map_510.plot(axes =ax[1], cmap = cmap, title = 'FOXSI 5-10 keV', norm = norm)

	foxsi_map_1025.plot(axes =ax[2], cmap = cmap, title = 'FOXSI 10-25 keV', norm = norm)

	plt.tight_layout()



def plot_foxsi_2(cmap = 'viridis', norm = Normalize()):
	fig, ax = plt.subplots(1,3, figsize = (15, 5))

	ax[0].imshow(foxsi_35['data'][0], origin = 'lower', extent = [xa[0], xa[1], ya[0], ya[1]], norm = norm, cmap = cmap)
	ax[0].set_title('FOXSI 3-5 keV')
	ax[0].set_xlabel('Solar X (arcsec)')
	ax[0].set_ylabel('Solar Y (arcsec)')

	ax[1].imshow(foxsi_510['data'][0], origin = 'lower', extent = [xa[0], xa[1], ya[0], ya[1]], norm = norm, cmap = cmap)
	ax[1].set_title('FOXSI 5-10 keV')
	ax[1].set_xlabel('Solar X (arcsec)')
	ax[1].set_ylabel('Solar Y (arcsec)')

	ax[2].imshow(foxsi_1025['data'][0], origin = 'lower', extent = [xa[0], xa[1], ya[0], ya[1]], norm = norm, cmap = cmap)
	ax[2].set_title('FOXSI 10-25 keV')
	ax[2].set_xlabel('Solar X (arcsec)')
	ax[2].set_ylabel('Solar Y (arcsec)')

	plt.tight_layout()

aia = readsav(sim_aia_file)

aia = readsav(sim_aia_file)
aia_meta = { 'crval1': aia['x'][209],
             'crval2': aia['y'][209],
             'cunit1': xunits,
             'cunit2':yunits,
             'cdelt1': aia['x'][1] - aia['x'][0],
		 	 'cdelt2': aia['y'][1] - aia['y'][0]
                
}


ye = map.Map(aia['syn131'], aia_meta)
#ye.plot(norm = LogNorm())



aia_131 = map.Map(aia['syn131'], aia_meta)
aia_171 = map.Map(aia['syn171'], aia_meta)
aia_193 = map.Map(aia['syn193'], aia_meta)
aia_211 = map.Map(aia['syn211'], aia_meta)
aia_335 = map.Map(aia['syn335'], aia_meta)
aia_94 = map.Map(aia['syn94'], aia_meta)


aia171 = aia['syn171']
aia131 = aia['syn131']
aia211 = aia['syn211']
aia94 = aia['syn94']
aia335 = aia['syn335']
aia193 = aia['syn193']

aia_171_small = imresize(aia171, (67,67))
aia_131_small = imresize(aia131, (67,67))
aia_211_small = imresize(aia211, (67,67))
aia_94_small = imresize(aia94, (67,67))
aia_335_small = imresize(aia335, (67,67))
aia_193_small = imresize(aia193, (67,67))

def plot_aia():

	fig, ax = plt.subplots(2, 3, figsize = (12, 8))
	cmap = 'magma'
	aia_171.plot(axes = ax[0][0], cmap= cmap, title = 'AIA 171 $\mathrm{\AA}$')#, norm = LogNorm())
	aia_193.plot(axes = ax[0][1], cmap= cmap, title = 'AIA 193 $\mathrm{\AA}$')#,norm = LogNorm())
	aia_211.plot(axes = ax[0][2], cmap= cmap, title = 'AIA 211 $\mathrm{\AA}$')#,norm = LogNorm())
	aia_335.plot(axes = ax[1][0], cmap= cmap, title = 'AIA 335 $\mathrm{\AA}$')#,norm = LogNorm())
	aia_94.plot(axes = ax[1][1], cmap= cmap, title = 'AIA 94 $\mathrm{\AA}$')#,norm = LogNorm())
	aia_131.plot(axes = ax[1][2], cmap= cmap, title = 'AIA 131 $\mathrm{\AA}$')#,norm = LogNorm())

	#for i in range(0, 2):
		#for j in range(0, 3):
			#ax[i][j].contour(foxsi_map_1025.data, levels = levels_1025, extent = [xa[0], xa[1], ya[0], ya[-1]], colors = 'red')
			#ax[i][j].contour(foxsi_map_510.data, levels = levels_510, extent = [xa[0], xa[1], ya[0], ya[-1]], colors = 'green')
			#ax[i][j].contour(foxsi_map_35.data, levels = levels_35, extent = [xa[0], xa[1], ya[0], ya[-1]], colors = 'blue')


	plt.tight_layout()


levels_35 = np.linspace(0.1*np.max(foxsi_map_35.data), np.max(foxsi_map_35.data), 30)

levels_510 = np.linspace(0.05*np.max(foxsi_map_510.data), np.max(foxsi_map_510.data), 30)

levels_1025 = np.linspace(0.05*np.max(foxsi_map_1025.data), np.max(foxsi_map_1025.data), 30)


def plot_aia2():

	fig, ax = plt.subplots(2, 3, figsize = (12, 8))
	cmap = 'viridis'
	ax[0][0].imshow(aia['syn171'], cmap= 'sdoaia171', norm = LogNorm(), extent = [xa[0], xa[1], ya[0], ya[-1]], origin = 'lower')
	ax[0][0].set_title('AIA 171 $\mathrm{\AA}$')
	ax[0][1].imshow(aia['syn193'], cmap= 'sdoaia193', norm = LogNorm(), extent = [xa[0], xa[1], ya[0], ya[-1]], origin = 'lower')
	ax[0][1].set_title('AIA 193 $\mathrm{\AA}$')
	ax[0][2].imshow(aia['syn211'], cmap= 'sdoaia211', norm = LogNorm(), extent = [xa[0], xa[1], ya[0], ya[-1]], origin = 'lower')
	ax[0][2].set_title('AIA 211 $\mathrm{\AA}$')
	ax[1][0].imshow(aia['syn335'], cmap= 'sdoaia335', norm = LogNorm(), extent = [xa[0], xa[1], ya[0], ya[-1]], origin = 'lower')
	ax[1][0].set_title('AIA 335 $\mathrm{\AA}$')
	ax[1][1].imshow(aia['syn94'], cmap= 'sdoaia94', norm = LogNorm(), extent = [xa[0], xa[1], ya[0], ya[-1]], origin = 'lower')
	ax[1][1].set_title('AIA 94 $\mathrm{\AA}$')
	ax[1][2].imshow(aia['syn131'], cmap= 'sdoaia131', norm = LogNorm(), extent = [xa[0], xa[1], ya[0], ya[-1]], origin = 'lower')
	ax[1][2].set_title('AIA 131 $\mathrm{\AA}$')

	'''for i in range(0, 2):
		for j in range(0, 3):
			ax[i][j].contour(foxsi_map_1025.data, levels = levels_1025, extent = [xa[0], xa[1], ya[0], ya[-1]], colors = 'red')
			ax[i][j].contour(foxsi_map_510.data, levels = levels_510, extent = [xa[0], xa[1], ya[0], ya[-1]], colors = 'green')
			ax[i][j].contour(foxsi_map_35.data, levels = levels_35, extent = [xa[0], xa[1], ya[0], ya[-1]], colors = 'blue')'''


	plt.tight_layout()



def plot_aia4():

	fig, ax = plt.subplots(2, 3, figsize = (12, 8))
	cmap = 'viridis'
	ax[0][0].imshow(aia['syn171'], cmap= 'sdoaia171', 
		extent = [xa[0], xa[1], ya[0], ya[-1]], origin = 'lower', norm = LogNorm())
	ax[0][0].set_title('AIA 171 $\mathrm{\AA}$')
	ax[0][0].set_xlabel('Solar X (arcsec)')
	ax[0][0].set_ylabel('Solar X (arcsec)')


	ax[0][1].imshow(aia['syn193'], cmap= 'sdoaia193', 
		extent = [xa[0], xa[1], ya[0], ya[-1]], origin = 'lower',  norm = LogNorm())
	ax[0][1].set_title('AIA 193 $\mathrm{\AA}$')
	ax[0][1].set_xlabel('Solar X (arcsec)')
	ax[0][1].set_ylabel('Solar X (arcsec)')

	ax[0][2].imshow(aia['syn211'], cmap= 'sdoaia211', 
		extent = [xa[0], xa[1], ya[0], ya[-1]], origin = 'lower', norm = LogNorm())
	ax[0][2].set_title('AIA 211 $\mathrm{\AA}$')
	ax[0][2].set_xlabel('Solar X (arcsec)')
	ax[0][2].set_ylabel('Solar X (arcsec)')

	ax[1][0].imshow(aia['syn335'], cmap= 'sdoaia335', 
		extent = [xa[0], xa[1], ya[0], ya[-1]], origin = 'lower', norm = LogNorm())
	ax[1][0].set_title('AIA 335 $\mathrm{\AA}$')
	ax[1][0].set_xlabel('Solar X (arcsec)')
	ax[1][0].set_ylabel('Solar X (arcsec)')

	ax[1][1].imshow(aia['syn94'], cmap= 'sdoaia94', 
	 extent = [xa[0], xa[1], ya[0], ya[-1]], origin = 'lower',  norm = LogNorm())
	ax[1][1].set_title('AIA 94 $\mathrm{\AA}$')
	ax[1][1].set_xlabel('Solar X (arcsec)')
	ax[1][1].set_ylabel('Solar X (arcsec)')


	ax[1][2].imshow(aia['syn131'], cmap= 'sdoaia131', 
		extent = [xa[0], xa[1], ya[0], ya[-1]], origin = 'lower', norm = LogNorm())
	ax[1][2].set_title('AIA 131 $\mathrm{\AA}$')
	ax[1][2].set_xlabel('Solar X (arcsec)')
	ax[1][2].set_ylabel('Solar X (arcsec)')

	'''for i in range(0, 2):
		for j in range(0, 3):
			ax[i][j].contour(foxsi_map_1025.data, levels = levels_1025, extent = [xa[0], xa[1], ya[0], ya[-1]], colors = 'red')
			ax[i][j].contour(foxsi_map_510.data, levels = levels_510, extent = [xa[0], xa[1], ya[0], ya[-1]], colors = 'green')
			ax[i][j].contour(foxsi_map_35.data, levels = levels_35, extent = [xa[0], xa[1], ya[0], ya[-1]], colors = 'blue')'''


	plt.tight_layout()

def plot_aia3():

	fig, ax = plt.subplots(2, 3, figsize = (12, 8))
	cmap = 'viridis'
	ax[0][0].imshow(aia['syn171'], cmap= 'sdoaia171', vmax = aia['syn171'].mean() + 3*aia['syn171'].std(), extent = [xa[0], xa[1], ya[0], ya[-1]], origin = 'lower')
	ax[0][0].set_title('AIA 171 $\mathrm{\AA}$')
	ax[0][0].set_xlabel('Solar X (arcsec)')
	ax[0][0].set_ylabel('Solar X (arcsec)')


	ax[0][1].imshow(aia['syn193'], cmap= 'sdoaia193', vmax = aia['syn193'].mean() + 3*aia['syn193'].std(), extent = [xa[0], xa[1], ya[0], ya[-1]], origin = 'lower')
	ax[0][1].set_title('AIA 193 $\mathrm{\AA}$')
	ax[0][1].set_xlabel('Solar X (arcsec)')
	ax[0][1].set_ylabel('Solar X (arcsec)')

	ax[0][2].imshow(aia['syn211'], cmap= 'sdoaia211', vmax = aia['syn211'].mean() + 3*aia['syn211'].std(), extent = [xa[0], xa[1], ya[0], ya[-1]], origin = 'lower')
	ax[0][2].set_title('AIA 211 $\mathrm{\AA}$')
	ax[0][2].set_xlabel('Solar X (arcsec)')
	ax[0][2].set_ylabel('Solar X (arcsec)')

	ax[1][0].imshow(aia['syn335'], cmap= 'sdoaia335', vmax = aia['syn335'].mean() + 3*aia['syn335'].std(), extent = [xa[0], xa[1], ya[0], ya[-1]], origin = 'lower')
	ax[1][0].set_title('AIA 335 $\mathrm{\AA}$')
	ax[1][0].set_xlabel('Solar X (arcsec)')
	ax[1][0].set_ylabel('Solar X (arcsec)')

	ax[1][1].imshow(aia['syn94'], cmap= 'sdoaia94', vmax = aia['syn94'].mean() + 3*aia['syn94'].std(), extent = [xa[0], xa[1], ya[0], ya[-1]], origin = 'lower')
	ax[1][1].set_title('AIA 94 $\mathrm{\AA}$')
	ax[1][1].set_xlabel('Solar X (arcsec)')
	ax[1][1].set_ylabel('Solar X (arcsec)')


	ax[1][2].imshow(aia['syn131'], cmap= 'sdoaia131', vmax = aia['syn131'].mean() + 3*aia['syn131'].std(), extent = [xa[0], xa[1], ya[0], ya[-1]], origin = 'lower')
	ax[1][2].set_title('AIA 131 $\mathrm{\AA}$')
	ax[1][2].set_xlabel('Solar X (arcsec)')
	ax[1][2].set_ylabel('Solar X (arcsec)')

	'''for i in range(0, 2):
		for j in range(0, 3):
			ax[i][j].contour(foxsi_map_1025.data, levels = levels_1025, extent = [xa[0], xa[1], ya[0], ya[-1]], colors = 'red')
			ax[i][j].contour(foxsi_map_510.data, levels = levels_510, extent = [xa[0], xa[1], ya[0], ya[-1]], colors = 'green')
			ax[i][j].contour(foxsi_map_35.data, levels = levels_35, extent = [xa[0], xa[1], ya[0], ya[-1]], colors = 'blue')'''


	plt.tight_layout()



levels_35 = np.linspace(0.1*np.max(foxsi_map_35.data), 0.8*np.max(foxsi_map_35.data), 30)

levels_510 = np.linspace(0.05*np.max(foxsi_map_510.data), 0.8*np.max(foxsi_map_510.data), 30)

levels_1025 = np.linspace(0.05*np.max(foxsi_map_1025.data), 0.8*np.max(foxsi_map_1025.data), 30)



def plot_tog(aia_cha, cmapp):
	fig, ax = plt.subplots(1, 3, figsize = (15, 5))
	ax[0].imshow(aia[aia_cha], cmap= cmapp, vmax = aia[aia_cha].mean() + 2*aia[aia_cha].std(), extent = [xa[0], xa[1], ya[0], ya[-1]], origin = 'lower')
	ax[0].contour(foxsi_map_35.data, levels = levels_35, extent = [xa[0], xa[1], ya[0], ya[-1]], colors = 'red')
	ax[0].text(750, 330, 'FOXSI 3-5keV', color = 'r')

	ax[1].imshow(aia[aia_cha], cmap= cmapp, vmax = aia[aia_cha].mean() + 2*aia[aia_cha].std(), extent = [xa[0], xa[1], ya[0], ya[-1]], origin = 'lower')
	ax[1].contour(foxsi_map_510.data, levels = levels_510, extent = [xa[0], xa[1], ya[0], ya[-1]], colors = 'blue')
	ax[1].text(750, 330, 'FOXSI 5-10keV', color = 'b')

	ax[2].imshow(aia[aia_cha], cmap= cmapp, vmax = aia[aia_cha].mean() + 2*aia[aia_cha].std(), extent = [xa[0], xa[1], ya[0], ya[-1]], origin = 'lower')
	ax[2].contour(foxsi_map_1025.data, levels = levels_1025, extent = [xa[0], xa[1], ya[0], ya[-1]], colors = 'green')
	ax[2].text(750, 330, 'FOXSI 10-25keV', color = 'g')


test = imresize(aia['syn171'], (67, 67))


xa = [aia['x'][0], aia['x'][-1]]
ya = [aia['y'][0], aia['y'][-1]]

def dem_plot(cmap = 'gnuplot'):
    k = 0
    fig, ax = plt.subplots(4, 7, figsize = (13,8))
    for i in range(0, 4):
        for j in range(0, 7):
            
            ax[i][j].imshow(aia['dem'][k], origin = 'lower', cmap = cmap, extent = [xa[0], xa[1], ya[0], ya[-1]] )
            ax[i][j].set_title(str(round(aia['t'][k]/1e6, 3))+' MK')
            ax[i][j].tick_params(labelleft = 'off', labelbottom = 'off')
            k = k + 1
    plt.tight_layout()





levels_35 = np.linspace(0.25*np.max(foxsi_map_35.data), np.max(foxsi_map_35.data), 10)

levels_510 = np.linspace(0.1*np.max(foxsi_map_510.data), np.max(foxsi_map_510.data), 30)

levels_1025 = np.linspace(0.1*np.max(foxsi_map_1025.data), np.max(foxsi_map_1025.data), 20)

def dem_plot2(cmap = 'gnuplot'):
    k = 0
    fig, ax = plt.subplots(4, 7, figsize = (18,10), sharex = True, sharey = True)
    for i in range(0, 4):
        for j in range(0, 7):
            
            ax[i][j].imshow(aia['dem'][k], origin = 'lower', cmap = cmap, extent = [xa[0], xa[1], ya[0], ya[-1]],
             vmin = np.min(aia['dem'][k]), vmax = np.mean(aia['dem'][k]) + 6*np.std(aia['dem'][k])) 
            #ax[i][j].contour(foxsi_map_35.data, levels = levels_35, extent = [xa[0], xa[1], ya[0], ya[-1]], colors = 'red')
            ax[i][j].contour(foxsi_map_35.data, extent = [xa[0], xa[1], ya[0], ya[-1]], colors = 'red')
            ax[i][j].set_title(str(round(aia['tm'][k]/1e6, 3))+' MK')
            
           # ax[i][j].tick_params(labelleft = 'off', labelbottom = 'off')
            k = k+1
    #ax[3][0].tick_params(labelleft = 'on', labelbottom = 'on')

    ax[3][0].set_xlabel('Arcsec')
    ax[3][0].set_ylabel('Arcsec')     
    plt.tight_layout()







