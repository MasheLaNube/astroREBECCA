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
	mean_flux= sec.flux.mean().value
	time_values= sec.time.jd #######################FIX
	flux_values= sec.flux.value

	print(f"Range of time: {inicio} - {fin}\n")
	print(f"Transit midpoint T0: {T0}\n")

	#Calculamos el número de períodos que deben restarse para que caiga en el rango
	n_ajuste= np.floor((inicio - T0) / P)
	T0_ajustado= T0 + n_ajuste * P

	print(f"T0 adjustes to range: {T0_ajustado}\n")

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
	print('Transit times: ', transitos)
	plt.show()
	
	
	
def fold_second_tr(sec, T0, P, PATH, binned, name=''):
	ax = sec.fold(period = P, epoch_time = T0+P/2).scatter()
	sec.fold(period = P, epoch_time = T0+P/2).bin(time_bin_size=binned).scatter(ax=ax,marker = 'o', s = 2, ec='magenta', fc = "none", lw=1, alpha=1)
	plt.title(name + ' Secondary Eclipse')
	plt.savefig(PATH + "/Secondary_eclp_"+ name +".png")
	plt.show()



def save_data(sec, T0, P, PATH, binned, name= ''):
	"""
	Obtain the phase plot doing phase folding. Saves phase folding data and "raw" data 
	"""
	print('Obtaining Phase folding...')
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
    
    
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii

def loading_data(PATH, binned, name, transitos, T0=None, P=None):
    """
    Load the data and apply phase folding or binning depending on the file type:
    - '_PF.csv': Data is already folded, only binning is applied.
    - '_NoPF.csv': Data is not folded, folding and transit marking is applied.
    """
    # Cargar los datos
    datapd = ascii.read(PATH + '/' + name)

    # Obtener las columnas relevantes
    data_time = datapd['time']
    data_flux = datapd['flux']
    data_err_flux = datapd['flux_err']

    plt.rcParams['figure.figsize'] = (8, 4)
    plt.rcParams['figure.dpi'] = 200

    mean_flux = data_flux.mean()
    ax0 = plt.subplot()

    if name.endswith('_PF.csv'):
        # Caso en el que los datos ya están en phase folding
        plt.plot(data_time, data_flux, '.k', markersize=0.5, zorder=1)

        # Bin the data manually
        bins = np.arange(data_time.min(), data_time.max(), binned)
        binned_flux, bin_edges = np.histogram(data_time, bins=bins, weights=data_flux)
        bin_counts, _ = np.histogram(data_time, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        binned_flux = np.divide(binned_flux, bin_counts, where=bin_counts != 0)  # Evitar división por 0

        # Plot the binned data
        ax0.scatter(bin_centers, binned_flux, color='orange', s=10, lw=0.01, zorder=2)
        plt.xlabel('Phase (Days)')
        plt.ylabel('Relative flux')

    elif name.endswith('_NoPF.csv'):
        # Caso en el que los datos son puros (no phase folding), aplicamos el gráfico sin folding
        for i in transitos:
            ax0.plot(data_time, data_flux, '.k', markersize=0.5, zorder=1)
            plt.plot(i, mean_flux, marker='x', color='r', markersize=1, zorder=2)

        ax0.set_xlabel("Time - 2457000 [BJTD days]")
        ax0.set_ylabel("Flux")
    
    plt.show()

	
		


def select_transit(PATH, bins, name, transitos):
    """
    To identify if the transits can be seen and save them if the user wishes.
    """
    datapd= ascii.read(PATH + '/' + name)
    i= 1  

    # Seleccionar el rango de datos
    for transit in transitos:
        start_time= transit - 1
        end_time= transit + 1

        # Filtrar los datos dentro del rango
        filtered_data= datapd[(datapd['time'] >= start_time) & (datapd['time'] <= end_time)]
        # Extraer las columnas de interés
        time_data= filtered_data['time']
        flux_data= filtered_data['flux']

        num_bins= int((time_data.max() - time_data.min()) / bins)
        # Realizar el binning con numpy
        bin_edges= np.linspace(time_data.min(), time_data.max(), num_bins + 1)  # Bordes de los bins
        bin_indices= np.digitize(time_data, bin_edges)  # Clasificar cada valor de tiempo en un bin
        bin_means_time= [time_data[bin_indices == i].mean() for i in range(1, num_bins + 1)]
        bin_means_flux= [flux_data[bin_indices == i].mean() for i in range(1, num_bins + 1)]

        # Crear el plot
        plt.figure(figsize=(10, 6))
        plt.scatter(time_data, flux_data, label='Flux')
        plt.scatter(bin_means_time, bin_means_flux, color='red', label='Binned Flux', s=50)
        plt.xlabel('Time - 2457000 [BTJD]')
        plt.ylabel('Flux')
        plt.title(f'Flux vs Time for Transit {i}')
        plt.grid(True)
        plt.legend()
        plt.show()

        # Mantener un ciclo para asegurar que el input sea válido
        while True:
            tr = input('Do you wanna save this transit? (y/n): ')
            if tr.lower() in ['y', 'yes']:
                # Guardar el tránsito con un nombre único basado en el contador 'i'
                df_to_save = pd.DataFrame({
                    'time': time_data,
                    'flux': flux_data
                })
                df_to_save.to_csv(f"{PATH}/Transit_{name}_transit_{i}.csv", index=False)
                print(f'Transit {i} saved as Transit_{name}_transit_{i}.csv.')
                i += 1  # Incrementar el contador solo si se guarda o se omite correctamente
                break  # Salir del ciclo de input válido
            elif tr.lower() in ['n', 'no']:
                print('Plot skipped.')
                i += 1  # Incrementar el contador solo si se guarda o se omite correctamente
                break  # Salir del ciclo de input válido
            else:
                print('Not a valid input. Please try again.')

