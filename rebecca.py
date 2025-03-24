import numpy as np 
import pandas as pd
import lightkurve as lk 
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, clear_output
from lightkurve.correctors import RegressionCorrector, DesignMatrix
from lightkurve.correctors import PLDCorrector
import warnings
import os
import time
import datetime
import astropy.io.ascii as ascii
from scipy.stats import binned_statistic
import webbrowser
	
def remove_outliers(table, column='flux', sigma_lower=5, sigma_upper=5):
	data = table[column]
	#Calcular la media y desviación estándar
	mean = np.mean(data)
	std = np.std(data)
	#Crear máscara para filtrar los valores atípicos
	mask = (data > mean - sigma_lower * std) & (data < mean + sigma_upper * std)
	# Devolver solo los datos que pasen el filtro
	return table[mask]
	    
    
def transit_identif(sec, T0, P, transits):
    """
    dentify when the transits will occur
    """
    #Convertimos los tiempos del Sector a formato numérico (en días julianos)
    inicio= sec.time.min().value
    fin= sec.time.max().value
    mean_flux= sec.flux.mean().value
    time_values= sec.time.jd 
    flux_values= sec.flux.value

    print(f"Range of time: {inicio} - {fin}\n")
    print(f"Transit midpoint T0: {T0}\n")

    #Calculamos el número de períodos que deben restarse para que caiga en el rango
    n_ajuste= np.floor((inicio - T0) / P)
    T0_ajustado= T0 + n_ajuste * P

    print(f"T0 adjustes to range: {T0_ajustado}\n")

    #Calculamos los tránsitos
    transitos= transits
    n= 0
    while True:
        transito= T0_ajustado + n * P  
        if transito >= inicio:
            transitos.append(transito)
        if transito > fin:
            break
        n += 1
    print('Transit times: ', transitos)
	
	




def save_data(sec, T0, P, PATH, binned, name= ''):
    """
    Obtain the phase plot doing phase folding. Saves phase folding data and "raw" data 
    """
    print('Obtaining Phase folding...')
    fig, (ax, fullplt_) = plt.subplots(nrows=2, figsize=(8, 6))

    a= sec.normalize().fold(period = P, epoch_time = T0)
    b = sec.normalize().flatten()

    ax = a.scatter(ax=ax)  
    fase = a.bin(time_bin_size=binned).scatter(ax=ax, color='orange', lw=1, zorder=2)
    ax.set_xlabel("Phase")
    ax.set_ylabel("Flux (Folded)")
    ax.set_title("Phase Folded Lightcurve")

    fullplt_ = b.scatter(ax=fullplt_)
    fullplt = b.bin(time_bin_size=binned).scatter(ax=fullplt_, color='orange', lw=1, zorder=2)
    fullplt_.set_xlabel("Time (BJTD days)")
    fullplt_.set_ylabel("Flux (Unfolded)")
    fullplt_.set_title("Raw Lightcurve")

    # Ajustar los gráficos para mejor visualización
    plt.tight_layout()

    plt.show()


    
    print('Saving data NORMALIZED...\n\n\n')
    a.to_csv(PATH +'/Data_' + name + '_PF.csv', overwrite=True)
    print('Data_' + name + '_PF.csv SAVED in ', PATH, ' folder.')
    sec.to_csv(PATH +'/Data_' + name + '_NoPF.csv', overwrite=True)
    print('Data_' + name + '_NoPF.csv SAVED in ', PATH, ' folder.')
    
    


def loading_data(PATH, binned, name, transitos):
    """
    Load the data and apply phase folding or binning depending on the file type:
    - '_PF.csv': Data is already folded, only binning is applied.
    - '_NoPF.csv': Data is not folded.

    Returns:
    - data_time: time data extracted from the file
    - data_flux: flux data extracted from the file
    - data_err_flux: error in flux data extracted from the file (if available)
    - datapd: the complete table as a DataFrame
    """
    # Load data
    data= ascii.read(PATH + '/' + name) 

    # Extract specific columns
    data_time= data['time']
    data_flux= data['flux']
    data_err_flux= data['flux_err']  # Use .get() to avoid key errors if column is missing

    # Plotting logic (for visualization)
    plt.rcParams['figure.figsize'] = (8, 4)
    plt.rcParams['figure.dpi'] = 200
    ax0= plt.subplot()

    if name.endswith('_PF.csv'):
        plt.plot(data_time, data_flux, '.k', markersize=0.5, zorder=1, label= 'Data')
        num_bins = binned
        bin_edges = np.linspace(data_time.min(), data_time.max(), num_bins + 1)
        bin_medians, _, _ = binned_statistic(data_time, data_flux, statistic='median', bins=bin_edges)
        bin_counts, _, _ = binned_statistic(data_time, data_flux, statistic='count', bins=bin_edges)
        valid_bins = bin_counts > 0
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#        valid_bins = ~np.isnan(bin_medians)
        ax0.scatter(bin_centers[valid_bins], bin_medians[valid_bins], color='orange', zorder=2, s=1, label= 'Binning')
        ax0.set_xlabel("Phase [JD]")
        ax0.set_ylabel("Normalized Flux")

    elif name.endswith('_NoPF.csv'):
        a=0
        for i in transitos:
            a+=1
            plt.plot(i, data_flux.mean(), marker='x', color='r', markersize=3, zorder=3, label=f'Transit midpoint {a}')
        ax0.plot(data_time, data_flux, '.k', markersize=0.5, zorder=1, label='Data')
        num_bins = binned
        bin_edges = np.linspace(data_time.min(), data_time.max(), num_bins + 1)
        bin_medians, _, _ = binned_statistic(data_time, data_flux, statistic='median', bins=bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_counts, _, _ = binned_statistic(data_time, data_flux, statistic='count', bins=bin_edges)
        valid_bins = (bin_counts > 0) & ~np.isnan(bin_medians)
#        valid_bins = ~np.isnan(bin_medians)
        ax0.scatter(bin_centers[valid_bins], bin_medians[valid_bins], color='orange', zorder=2, s=1, label='Binning')
        ax0.set_xlabel("Time - 2457000 [JD days]")
        ax0.set_ylabel("Normalized Flux")
        
    plt.legend(prop={'size': 8}, loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()

    # Return the desired values
    return data_time, data_flux, data_err_flux

def GUI(PATH, name):
    """
    Load the data and apply phase folding or binning depending on the file type:
    - '_PF.csv': Data is already folded, only binning is applied.
    - '_NoPF.csv': Data is not folded.

    Returns:
    - data_time: time data extracted from the file
    - data_flux: flux data extracted from the file
    - data_err_flux: error in flux data extracted from the file (if available)
    - datapd: the complete table as a DataFrame
    """
    # Load data
    data= ascii.read(PATH + '/' + name)  # Converts to DataFrame
    
    
    # Extract specific columns
    data_time= data['time']
    data_flux= data['flux']
    

#Crear múltiples versiones del gráfico con diferentes números de bins
    bins_values= list(range(10, 201, 10))  #Bins desde 10 hasta 200
    frames= []
    
    for num_bins in bins_values:
        bin_edges= np.linspace(data_time.min(), data_time.max(), num_bins + 1)
        bin_medians, _, _ = binned_statistic(data_time, data_flux, statistic='median', bins=bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        valid_bins = ~np.isnan(bin_medians)

        #Asegurar que el binned data esté arriba
        frame = go.Frame(
            data=[
                go.Scatter(x=data_time, y=data_flux, mode='markers', marker=dict(size=2, opacity=0.5), name='Raw Data'),
                go.Scatter(x=bin_centers[valid_bins], y=bin_medians[valid_bins], 
                           mode='markers', marker=dict(color='orange', size=6), name='Binned Data')
            ],
            name=str(num_bins)
        )
        frames.append(frame)

    # Crear la figura base con los datos sin binning
    fig= go.Figure()

    fig.add_trace(go.Scatter(x=data_time, y=data_flux, mode='markers', marker=dict(size=2, opacity=0.5), name='Raw Data'))

    # Agregar la primera versión binned
    first_bin= bins_values[0]
    bin_edges= np.linspace(data_time.min(), data_time.max(), first_bin + 1)
    bin_medians, _, _ = binned_statistic(data_time, data_flux, statistic='median', bins=bin_edges)
    bin_centers= (bin_edges[:-1] + bin_edges[1:]) / 2
    valid_bins= ~np.isnan(bin_medians)

    fig.add_trace(go.Scatter(x=bin_centers[valid_bins], y=bin_medians[valid_bins], 
                             mode='markers', marker=dict(color='orange', size=6), name='Binned Data'))

    # Agregar frames para animación
    fig.update(frames=frames)

    # Agregar slider de Plotly
    fig.update_layout(
        sliders=[{
            "steps": [
                {"method": "animate", "args": [[str(b)], {"frame": {"duration": 0}, "mode": "immediate"}], "label": str(b)}
                for b in bins_values
            ],
            "currentvalue": {"prefix": "Bins: "}
        }],
        title='Lightcurve',
        xaxis_title='Time [JD]',
        yaxis_title='Normalized Flux',
        template='plotly_dark'
    )

    # Forzar actualización de trazas para asegurar que binned data esté encima
    fig.update_traces(overwrite=True)

    fig.show()

    # Preguntar si se desea guardar
    save = input('Do you want to save this plot?[y/n]: ')
    if save.lower() == 'y':
        pio.write_html(fig, PATH+'/'+f"{name}_interactive.html")
        print(f"Archivo guardado como {name}_interactive.html")





'''
					In development
'''


def select_transit(PATH, bins, name, transitos):
    """
    To identify if the transits can be seen and save them if the user wishes.
    Returns a list of light curves (LCs), each as an array with columns: time, flux, flux_err.
    """
    # Load data and convert to Pandas DataFrame for easier handling
    datapd = ascii.read(PATH + '/' + name)
    datapd = remove_outliers(datapd, column='flux', sigma_lower=5, sigma_upper=5)
    data_time = datapd['time']
    data_flux = datapd['flux']
    data_err_flux = datapd['flux_err']  # Defaulting to 0 if 'flux_err' is not present

    lc_list = []  # List to store the light curves

    # Select data within the range of each transit
    i = 1
    for transit in transitos:
        start_time = transit - 1
        end_time = transit + 1

        # Filter data within the specified time range
        filtered_data = datapd[(datapd['time'] >= start_time) & (datapd['time'] <= end_time)]

        time_data = filtered_data['time']
        flux_data = filtered_data['flux']
        flux_err_data = filtered_data['flux_err']
        
        # Bin the data
        num_bins = int((time_data.max() - time_data.min()) / bins)
        bin_edges = np.linspace(time_data.min(), time_data.max(), num_bins + 1)
        bin_indices = np.digitize(time_data, bin_edges)
        bin_means_time = [time_data[bin_indices == j].mean() for j in range(1, num_bins + 1)]
        bin_means_flux = [flux_data[bin_indices == j].mean() for j in range(1, num_bins + 1)]
        bin_means_err_flux = [flux_err_data[bin_indices == j].mean() for j in range(1, num_bins + 1)]

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.scatter(time_data, flux_data, label='Flux')
        plt.scatter(bin_means_time, bin_means_flux, color='red', label='Binned Flux', s=50)
        plt.xlabel('Time - 2457000 [BTJD]')
        plt.ylabel('Flux')
        plt.title(f'Flux vs Time for Transit {i}')
        plt.grid(True)
        plt.legend()
        plt.show()

        # Collect the light curve data into an array and store it in the list
        lc_array = np.array([time_data, flux_data, flux_err_data]).T
        lc_list.append(lc_array)

        # Prompt user for saving the transit
        while True:
            tr = input('Do you wanna save this transit? (y/n): ')
            if tr.lower() in ['y', 'yes']:
                # Save the transit data using numpy
                save_array = np.array([time_data, flux_data, flux_err_data]).T  # Transpose to get columns
                header = 'time,flux,flux_err'  # Header for the CSV file
                np.savetxt(f"{PATH}/Transit_{name}_transit_{i}.csv", save_array, delimiter=',', header=header, comments='', fmt='%f')
                print(f'Transit {i} saved as Transit_{name}_transit_{i}.csv.')
                i += 1
                break
            elif tr.lower() in ['n', 'no']:
                print('Plot skipped.')
                i += 1
                break
            else:
                print('Not a valid input. Please try again.')

    return lc_list

