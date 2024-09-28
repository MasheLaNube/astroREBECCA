import numpy as np 
import lightkurve as lk 
import matplotlib.pyplot as plt 
from lightkurve.correctors import RegressionCorrector, DesignMatrix
from lightkurve.correctors import PLDCorrector
import warnings
import pandas as pd
import os
import datetime
import astropy.io.ascii as ascii

def rem_outliers(sec, PATH, Binned, name=''):
	"""
	Remove Outliers from all the data
	"""

	sec= sec.remove_outliers(sigma_lower = 20, sigma_upper = 2.7)
	sec.plot()
	plt.title(name)
	plt.savefig(PATH + "/" + name, dpi=300)
	ax= sec.plot()
	sec.bin(time_bin_size=Binned).plot(ax=ax,color='orange',lw=0.5)
	plt.title(name + ' binned')
	    
    
def transit_identif(sec, T0, P, transits):
	"""
	Identify when the transits will occur
	"""
	#Convertimos los tiempos del Sector a formato numérico (en días julianos)
	inicio= sec.time.min().value
	fin= sec.time.max().value

	print(f"Range of time: {inicio} - {fin}")
	print(f"Transit midpoint T0: {T0}")

	#Calculamos el número de períodos que deben restarse para que caiga en el rango
	n_ajuste= np.floor((inicio - T0) / P)
	T0_ajustado= T0 + n_ajuste * P

	print(f"T0 adjustes to range: {T0_ajustado}")

	# Calculamos los tránsitos
	transitos= transits
	n= 0
	while True:
		transito= T0_ajustado + n * P
		if transito > fin:
		    break
		if transito >= inicio:
		    transitos.append(transito)
		n += 1
	# Graficamos los tránsitos si existen
	if transitos:
		plt.scatter(transitos, np.zeros_like(transitos), marker='x', color='red')
		plt.xlabel('Time - 2457000 [BTJD]')
		plt.ylabel('')
		plt.title('Time of Tránsits')
		plt.grid(True)
	print('Transit times: ', transitos)
	plt.show()
    


def save_data(sec, T0, P, PATH, binned,name= ''):
	"""
	Obtain the phase plot doing phase folding. Saves phase folding data and "raw" data 
	"""
	ax= sec.fold(period = P, epoch_time = T0, normalize_phase=False).scatter()
	a= sec.fold(period = P, epoch_time = T0)
	fase= sec.fold(period = P, epoch_time = T0).bin(time_bin_size=binned).scatter(ax=ax,color='orange',lw=1)

	plt.savefig(PATH + "/" + name)
	plt.show()
	print('Saving data...\n\n\n')
	a.to_csv(PATH +'/Data_' + name + '_PF.csv', overwrite=True)
	print('Data_' + name + '_PF.csv SAVED in ', PATH, ' folder.')
	sec.to_csv(PATH +'/Data_' + name + '_NoPF.csv', overwrite=True)
	print('Data_' + name + '_NoPF.csv SAVED in ', PATH, ' folder.')
    
    
def loading_data(PATH, name, transitos):
	datapd = ascii.read(PATH + '/' + name) 

	#Obtener las columnas relevantes
	data_time= datapd['time']
	data_flux= datapd['flux']
	data_err_flux= datapd['flux_err']
	nlc= datapd['flux'].shape[0]

	print(datapd)
	
	plt.rcParams['figure.figsize'] = 4,2
	plt.rcParams['figure.dpi'] = 200
	
	mean_flux= data_flux.mean()

	for i in transitos:
	    ax0= plt.subplot()
	    ax0.plot(data_time, data_flux, '.k', markersize=0.5, zorder= 1)
	    plt.plot(i, mean_flux, marker='x', color= 'r', markersize= 1, zorder= 2)
	    ax0.set_xlabel("Time from central transit")
	    ax0.set_ylabel("Relative flux")


def select_transit(PATH, bins, name, transitos):
	datapd= ascii.read(PATH + '/' + name) 
	#Seleccionar el rango de datos
	for transit in transitos:
		start_time= transit - 1
		end_time= transit + 1

		#Filtrar los datos dentro del rango
		filtered_data= datapd[(datapd['time'] >= start_time) & (datapd['time'] <= end_time)]
		#Extraer las columnas de interés
		time_data= filtered_data['time']
		flux_data= filtered_data['flux']
		num_bins = int((time_data.max() - time_data.min()) / bins)
		#Realizar el binning con numpy
		bin_edges = np.linspace(time_data.min(), time_data.max(), num_bins + 1)  # Bordes de los bins
		bin_indices = np.digitize(time_data, bin_edges)  # Clasificar cada valor de tiempo en un bin
		bin_means_time = [time_data[bin_indices == i].mean() for i in range(1, num_bins + 1)]
		bin_means_flux = [flux_data[bin_indices == i].mean() for i in range(1, num_bins + 1)]
		plt.figure(figsize=(10, 6))
		plt.scatter(time_data, flux_data, label='Flux')
		plt.scatter(bin_means_time, bin_means_flux, color='red', label='Binned Flux', s=50)
		plt.xlabel('Time - 2457000 [BTJD]')
		plt.ylabel('Flux')
		plt.title('Flux vs Time')
		plt.grid(True)
		plt.legend()
		plt.show()
		
		tr = input('Do you wanna save this transit? (y/n): ')
		if tr.lower() in ['y', 'yes']:
			df_to_save= pd.DataFrame({
				'time': time_data,
                		'flux': flux_data
               			})
			df_to_save.to_csv(PATH + '/Transit_' + name + '.csv', index=False)
			print('Plot saved.')
        		
		elif tr.lower() in ['n', 'no']:
			print('Plot skipped.')
			pass
				
		else:
			print('Not valid input. Try again')
		

