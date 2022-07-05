#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 10:07:34 2022

@author: Marco
"""
import os
import sys

import numpy as np
import xarray as xr
from numpy.polynomial import Polynomial


# TO DO: argomenti unwrap, dem, bordo, passo, pixdx, pixdy

# Load dei file di input
unwrap = xr.open_dataset("unwrap.grd", cache=False)
dem = xr.open_dataset("dem_cut.grd", cache=False)

# Con questo mergio le coordinate x e y per creare la dim "coord" e poter estrapolare gli array
unwrap_stacked = unwrap.stack(coord=["x", "y"])
# Estrapolo gli array
x = unwrap_stacked.coords["x"].values
y = unwrap_stacked.coords["y"].values
z = unwrap_stacked.z.values
# Creo un array di nan lungo quanto z per salvare i risultati della regressione
risultati_regress1 = np.full(z.size, np.nan)
risultati_regress2 = np.full(z.size, np.nan)
# Estrapolo la matrice da cui prendere gli indici
z_ind = unwrap.z.values
# Numnero complessivo di indici
ind = x.size

# grandezza del pixel in coordinate radar
pixdx = 16
pixdy = 4

# Definisco il gli indici da non considerare per il bordo e per metà finestra
bordo = 5
passo = 27
passox = pixdx * passo
passoy = pixdy * passo

# Indici da scartare dalle 4 coordinate (x min e max e y min e max)
step = bordo + passo

# Indici dell'unwrap. Sono uguali a quelli del dem
n, m = np.indices(z_ind.shape)

# Trovo gli indici dei vertici
mymin = step
mymax = (np.max(n) + 1) - step
nxmin = step
nxmax = (np.max(m) + 1) - step

# Metodo di suddivisione area: per indice
unwrap_sel = unwrap.isel(x=slice(nxmin, nxmax), y=slice(mymin, mymax))
unwrap_sel_stacked = unwrap_sel.stack(coord=["x", "y"])


# trovo il primo indice min e l'ultimo per max sull'array di x per costruire l'array della regressione
result_xmin_ind = np.where(x == unwrap_sel.x.values[0])
ind_xmin = np.min(result_xmin_ind)


result_xmax_ind = np.where(x == np.max(unwrap_sel.x.values))
ind_xmax = np.max(result_xmax_ind)


ind_full = np.full(ind, False)
ind_full[ind_xmin : ind_xmax + 1] = True
for k in range(ind):
    # ←print(k)
    if not ind_full[k]:
        continue
    if np.isnan(z[k]):
        continue
    # if z[k]==0:
    #     continue
    # trovo le coordinate del punto centrale
    xi = x[k]
    yi = y[k]
    # creo la finestra da centrare sul punto
    windxmin = xi - passox
    windxmax = xi + passox
    windymin = yi - passoy
    windymax = yi + passoy

    # seleziono l'area inbase alla finestra
    sottofinestra_unwrap = unwrap.sel(
        x=slice(windxmin, windxmax), y=slice(windymin, windymax)
    )
    # faccio lo stack per ottenere la coppia di coordinate per ogni valore della z
    sottofinestra_unwrap_stacked = sottofinestra_unwrap.stack(coord=["x", "y"])
    # estrapolo le coordinate dei punti e la z in formato array
    # x3= sottofinestra_unwrap_stacked.coords['x'].values
    # y3= sottofinestra_unwrap_stacked.coords['y'].values
    z_unwrap = sottofinestra_unwrap_stacked.z.values
    # trovo i nan nell'array della finestra dell'unwrap
    no_nan = np.isnan(z_unwrap)

    # faccio la stessa cosa per il dem
    sottofinestra_dem = dem.sel(
        x=slice(windxmin, windxmax), y=slice(windymin, windymax)
    )
    # faccio lo stack per ottenere la coppia di coordinate per ogni valore della z
    sottofinestra_dem_stacked = sottofinestra_dem.stack(coord=["x", "y"])
    # estrapolo le coordinate dei punti e la z in formato array
    # xdem= sottofinestra_dem_stacked.coords['x'].values
    # ydem= sottofinestra_dem_stacked.coords['y'].values
    z_dem = sottofinestra_dem_stacked.z.values

    # regressione. Deve essere fatta con due array senza nan
    regressione = Polynomial.fit(
        z_unwrap[not no_nan], z_dem[not no_nan], deg=1
    ).convert()
    # estrapolo i coefficienti uno dei quali da inserire nell'array della correzione
    coef1 = regressione.coef[0]
    coef2 = regressione.coef[1]
    # salvo i parametri su degli array delle stesse dimensioni dei file di input
    risultati_regress1[k] = coef1
# TO DO: risultati_regress0 e risultati_regress1 scritte a grd
