
# Importing numpy 
import numpy as np
# Importing Scipy 
import scipy as sp
# Importing Pandas Library 
import pandas as pd
# import glob function to scrap files path
from glob import glob
# import display() for better visualitions of DataFrames and arrays
from IPython.display import display
# import pyplot for plotting
import matplotlib.pyplot as plt
plt.style.use('bmh') # for better plots

import math 
############################################################################################

# df is dataframe contains 3 columns (3 axial signals X,Y,Z)

# mean
def mean_axial(df):
    array=np.array(df) # convert dataframe into 2D numpy array for efficiency
    mean_vector = list(array.mean(axis=0)) # calculate the mean value of each column
    return mean_vector # return mean vetor
# std
def std_axial(df):
    array=np.array(df)# convert dataframe into 2D numpy array for efficiency
    std_vector = list(array.std(axis=0))# calculate the standard deviation value of each column
    return std_vector
# mad
from statsmodels.robust import mad as median_deviation # import the median deviation function
def mad_axial(df):
    array=np.array(df)# convert dataframe into 2D numpy array for efficiency
    mad_vector = list(median_deviation(array,axis=0)) # calculate the median deviation value of each column
    return mad_vector
# max
def max_axial(df):
    array=np.array(df)# convert dataframe into 2D numpy array for efficiency
    max_vector=list(array.max(axis=0))# calculate the max value of each column
    return max_vector
# min
def min_axial(df):
    array=np.array(df)# convert dataframe into 2D numpy array for efficiency
    min_vector=list(array.min(axis=0))# calculate the min value of each column
    return min_vector
# IQR
from scipy.stats import iqr as IQR # import interquartile range function (Q3(column)-Q1(column))
def IQR_axial(df):
    array=np.array(df)# convert dataframe into 2D numpy array for efficiency
    IQR_vector=list(np.apply_along_axis(IQR,0,array))# calculate the inter quartile range value of each column
    return IQR_vector
# Entropy
from scipy.stats import entropy # import the entropy function
def entropy_axial(df):
    array=np.array(df)# convert dataframe into 2D numpy array for efficiency
    entropy_vector=list(np.apply_along_axis(entropy,0,abs(array)))# calculate the entropy value of each column
    return entropy_vector

# mag column : is one column contains one mag signal values
# same features mentioned above were calculated for each column

# mean
def mean_mag(mag_column):
    array=np.array(mag_column)
    mean_value = float(array.mean())
    return mean_value

# std: standard deviation of mag column
def std_mag(mag_column):
    array=np.array(mag_column)
    std_value = float(array.std()) # std value 
    return std_value
# mad: median deviation
def mad_mag(mag_column):
    array=np.array(mag_column)
    mad_value = float(median_deviation(array))# median deviation value of mag_column
    return mad_value
# max
def max_mag(mag_column):
    array=np.array(mag_column)
    max_value=float(array.max()) # max value 
    return max_value
# min
def min_mag(mag_column):
    array=np.array(mag_column)
    min_value= float(array.min()) # min value
    return min_value
# IQR
def IQR_mag(mag_column):
    array=np.array(mag_column)
    IQR_value=float(IQR(array))# Q3(column)-Q1(column)
    return IQR_value
# Entropy
def entropy_mag(mag_column):
    array=np.array(mag_column)
    entropy_value=float(entropy(array)) # entropy signal
    return entropy_value

# Functions used to generate time axial features

# df is dataframe contains 3 columns (3 axial signals X,Y,Z)
# sma
def t_sma_axial(df):
    array=np.array(df)
    sma_axial=float(abs(array).sum())/float(3) # sum of areas under each signal
    return sma_axial # return sma value

# energy
def t_energy_axial(df):
    array=np.array(df)
    energy_vector=list((array**2).sum(axis=0)) # energy value of each df column
    return energy_vector # return energy vector energy_X,energy_Y,energy_Z

# AR vector (auto regression coefficients from 1 to 4)

# define the arbugr function
#auto regression coefficients with using burg method with order from 1 to 4
# from spectrum import *

##############################################################################################
# I took this function as it is from this link ------>    https://github.com/faroit/freezefx/blob/master/fastburg.py
# This fucntion and the original function arburg in the library spectrum generate the same first 3 coefficients 
#for all windows the original burg method is low and for some windows it cannot generate all 4th coefficients 

def _arburg2(X, order):
    """This version is 10 times faster than arburg, but the output rho is not correct.
    returns [1 a0,a1, an-1]
    """
    x = np.array(X)
    N = len(x)

    if order == 0.:
        raise ValueError("order must be > 0")

    # Initialisation
    # ------ rho, den
    rho = sum(abs(x)**2.) / N  # Eq 8.21 [Marple]_
    den = rho * 2. * N

    # ------ backward and forward errors
    ef = np.zeros(N, dtype=complex)
    eb = np.zeros(N, dtype=complex)
    for j in range(0, N):  # eq 8.11
        ef[j] = x[j]
        eb[j] = x[j]

    # AR order to be stored
    a = np.zeros(1, dtype=complex)
    a[0] = 1
    # ---- rflection coeff to be stored
    ref = np.zeros(order, dtype=complex)

    E = np.zeros(order+1)
    E[0] = rho

    for m in range(0, order):
        # print m
        # Calculate the next order reflection (parcor) coefficient
        efp = ef[1:]
        ebp = eb[0:-1]
        # print efp, ebp
        num = -2. * np.dot(ebp.conj().transpose(), efp)
        den = np.dot(efp.conj().transpose(),  efp)
        den += np.dot(ebp,  ebp.conj().transpose())
        ref[m] = num / den

        # Update the forward and backward prediction errors
        ef = efp + ref[m] * ebp
        eb = ebp + ref[m].conj().transpose() * efp

        # Update the AR coeff.
        a.resize(len(a)+1)
        a = a + ref[m] * np.flipud(a).conjugate()

        # Update the prediction error
        E[m+1] = np.real((1 - ref[m].conj().transpose() * ref[m])) * E[m]
        # print 'REF', ref, num, den
    return a, E[-1], ref

#################################################################################################################

# to generate arburg (order 4) coefficents for 3 columns [X,Y,Z]
def t_arburg_axial(df):
    # converting signals to 1D numpy arrays for efficiency
    array_X=np.array(df[df.columns[0]])
    array_Y=np.array(df[df.columns[1]])
    array_Z=np.array(df[df.columns[2]])
    
    AR_X = list(_arburg2(array_X,4)[0][1:].real) # list contains real parts of all 4th coefficients generated from signal_X
    AR_Y = list(_arburg2(array_Y,4)[0][1:].real) # list contains real parts of all 4th coefficients generated from signal_Y
    AR_Z = list(_arburg2(array_Z,4)[0][1:].real) # list contains real parts of all 4th coefficients generated from signal_Z
    
    # selecting [AR1 AR2 AR3 AR4] real components for each axis concatenate them in one vector
    AR_vector= AR_X + AR_Y+ AR_Z
    
    
    # AR_vector contains 12 values 4values per each axis 
    return AR_vector


from scipy.stats import pearsonr

def t_corr_axial(df): # it returns 3 correlation features per each 3-axial signals in  time_window
    
    array=np.array(df)
    
    Corr_X_Y=float(pearsonr(array[:,0],array[:,1])[0]) # correlation value between signal_X and signal_Y
    Corr_X_Z=float(pearsonr(array[:,0],array[:,2])[0]) # correlation value between signal_X and signal_Z
    Corr_Y_Z=float(pearsonr(array[:,1],array[:,2])[0]) # correlation value between signal_Y and signal_Z
    
    corr_vector =[Corr_X_Y, Corr_X_Z, Corr_Y_Z] # put correlation values in list
    
    return corr_vector 

# Functions used to generate time magnitude features

# sma: signal magnitude area
def t_sma_mag(mag_column):
    array=np.array(mag_column)
    sma_mag=float(abs(array).sum())# signal magnitude area of one mag column
    return sma_mag

# energy
def t_energy_mag(mag_column):
    array=np.array(mag_column)
    energy_value=float((array**2).sum()) # energy of the mag signal
    return energy_value

# arburg: auto regression coefficients using the burg method
def t_arburg_mag(mag_column):
    
    array = np.array(mag_column)
    
    AR_vector= list(_arburg2(array,4)[0][1:].real) # AR1, AR2, AR3, AR4 of the mag column
    #print(AR_vector)
    return AR_vector

# Functions used to generate frequency axial features
# each df here is dataframe contains 3 columns (3 axial frequency domain signals X,Y,Z)
# signals were obtained from frequency domain windows
# sma
def f_sma_axial(df):
    array=np.array(df)
    sma_value=float((abs(array)/math.sqrt(128)).sum())/float(3) # sma value of 3-axial f_signals
    return sma_value

# energy
def f_energy_axial(df):
    array=np.array(df)

    # spectral energy vector
    energy_vector=list((array**2).sum(axis=0)/float(len(array))) # energy of: f_signalX,f_signalY, f_signalZ
    return energy_vector # enrgy veactor=[energy(signal_X),energy(signal_Y),energy(signal_Z)]


####### Max Inds and Mean_Freq Functions#######################################
# built frequencies list (each column contain 128 value)
# duration between each two successive captures is 0.02 s= 1/50hz
freqs=sp.fftpack.fftfreq(128, d=0.02) 
                                

# max_Inds
def f_max_Inds_axial(df):
    array=np.array(df)
    max_Inds_X =freqs[array[1:65,0].argmax()+1] # return the frequency related to max value of f_signal X
    max_Inds_Y =freqs[array[1:65,1].argmax()+1] # return the frequency related to max value of f_signal Y
    max_Inds_Z =freqs[array[1:65,2].argmax()+1] # return the frequency related to max value of f_signal Z
    max_Inds_vector= [max_Inds_X,max_Inds_Y,max_Inds_Z]# put those frequencies in a list
    return max_Inds_vector

# mean freq()
def f_mean_Freq_axial(df):
    array=np.array(df)
    
    # sum of( freq_i * f_signal[i])/ sum of signal[i]
    mean_freq_X = np.dot(freqs,array[:,0]).sum() / float(array[:,0].sum()) #  frequencies weighted sum using f_signalX
    mean_freq_Y = np.dot(freqs,array[:,1]).sum() / float(array[:,1].sum()) #  frequencies weighted sum using f_signalY 
    mean_freq_Z = np.dot(freqs,array[:,2]).sum() / float(array[:,2].sum()) #  frequencies weighted sum using f_signalZ
    
    mean_freq_vector=[mean_freq_X,mean_freq_Y,mean_freq_Z] # vector contain mean frequencies[X,Y,Z]
    
    return  mean_freq_vector

###################################################################################

########## Skewness & Kurtosis Functions #######################################
from scipy.stats import kurtosis       # kurtosis function
from scipy.stats import skew           # skewness function
    
def f_skewness_and_kurtosis_axial(df):
    array=np.array(df)
    
    skew_X= skew(array[:,0])  # skewness value of signal X
    kur_X= kurtosis(array[:,0])  # kurtosis value of signal X
    
    skew_Y= skew(array[:,1]) # skewness value of signal Y
    kur_Y= kurtosis(array[:,1])# kurtosis value of signal Y
    
    skew_Z= skew(array[:,2])# skewness value of signal Z
    kur_Z= kurtosis(array[:,2])# kurtosis value of signal Z
    
    skew_kur_3axial_vector=[skew_X,kur_X,skew_Y,kur_Y,skew_Z,kur_Z] # return the list
    
    return skew_kur_3axial_vector
##################################################################################


#################### Bands Energy FUNCTIONS ########################

# bands energy levels (start row,end_row) end row not included 
B1=[(1,9),(9,17),(17,25),(25,33),(33,41),(41,49),(49,57),(57,65)] 
B2=[(1,17),(17,31),(31,49),(49,65)]
B3=[(1,25),(25,49)]

def f_one_band_energy(f_signal,band): # f_signal is one column in frequency axial signals in f_window
    # band: is one tuple in B1 ,B2 or B3 
    f_signal_bounded = f_signal[band[0]:band[1]] # select f_signal components included in the band
    energy_value=float((f_signal_bounded**2).sum()/float(len(f_signal_bounded))) # energy value of that band
    return energy_value

def f_all_bands_energy_axial(df): # df is dataframe contain 3 columns (3-axial f_signals [X,Y,Z])
    
    E_3_axis =[]
    
    array=np.array(df)
    for i in range(0,3): # iterate throw signals
        E1=[ f_one_band_energy( array,( B1 [j][0], B1 [j][1]) ) for j in range(len(B1))] # energy bands1 values of f_signal
        E2=[ f_one_band_energy( array,( B2 [j][0], B2 [j][1]) ) for j in range(len(B2))]# energy bands2 values of f_signal
        E3=[ f_one_band_energy( array,( B3 [j][0], B3 [j][1]) ) for j in range(len(B3))]# energy bands3 values of f_signal
        E_one_axis = E1+E2+E3 # list of energy bands values of one f_signal
        E_3_axis= E_3_axis + E_one_axis # add values to the global list
    
    return E_3_axis

# sma
def f_sma_mag(mag_column):
    array=np.array(mag_column)
    sma_value=float((abs(array)/math.sqrt(len(mag_column))).sum()) # sma of one mag f_signals
    return sma_value

# energy
def f_energy_mag(mag_column):
    array=np.array(mag_column)
    # spectral energy value
    energy_value=float((array**2).sum()/float(len(array))) # energy value of one mag f_signals
    return energy_value


####### Max Inds and Mean_Freq Functions#######################################


# max_Inds
def f_max_Inds_mag(mag_column):
    array=np.array(mag_column)
    max_Inds_value =float(freqs[array[1:65].argmax()+1]) # freq value related with max component
    return max_Inds_value

# mean freq()
def f_mean_Freq_mag(mag_column):
    array=np.array(mag_column)
    mean_freq_value = float(np.dot(freqs,array).sum() / float(array.sum())) # weighted sum of one mag f_signal
    return  mean_freq_value

###################################################################################

########## Skewness & Kurtosis Functions #######################################

from scipy.stats import skew           # skewness
def f_skewness_mag(mag_column):
    array=np.array(mag_column)
    skew_value     = float(skew(array)) # skewness value of one mag f_signal
    return skew_value

from scipy.stats import kurtosis       # kurtosis
def f_kurtosis_mag(mag_column):
    array=np.array(mag_column)
    kurtosis_value = float(kurtosis(array)) # kurotosis value of on mag f_signal

    return kurtosis_value
##################################################################################

############### Angles Functions ####################################
from math import acos # inverse of cosinus function
from math import sqrt # square root function

########Euclidian magnitude 3D############
def magnitude_vector(vector3D): # vector[X,Y,Z]
    return sqrt((vector3D**2).sum()) # eulidian norm of that vector

###########angle between two vectors in radian ###############
def angle(vector1, vector2):
    vector1_mag=magnitude_vector(vector1) # euclidian norm of V1
    vector2_mag=magnitude_vector(vector2) # euclidian norm of V2
   
    scalar_product=np.dot(vector1,vector2) # scalar product of vector 1 and Vector 2
    cos_angle=scalar_product/float(vector1_mag*vector2_mag) # the cosinus value of the angle between V1 and V2
    
    # just in case some values were added automatically
    if cos_angle>1:
        cos_angle=1
    elif cos_angle<-1:
        cos_angle=-1
    
    angle_value=float(acos(cos_angle)) # the angle value in radian
    return angle_value # in radian.

################## angle_features ############################
def angle_features(t_window): # it returns 7 angles per window
    angles_list=[]# global list of angles values

    # mean value of each column t_body_acc[X,Y,Z]
    V2_columns=['t_grav_acc_X','t_grav_acc_Y','t_grav_acc_Z']
    V2_Vector=np.array(t_window[V2_columns].mean()) # mean values
    
    # angle 0: angle between (t_body_acc[X.mean,Y.mean,Z.mean], t_gravity[X.mean,Y.mean,Z.mean])
    V1_columns=['t_body_acc_X','t_body_acc_Y','t_body_acc_Z']
    V1_Vector=np.array(t_window[V1_columns].mean()) # mean values of t_body_acc[X,Y,Z]
    angles_list.append(angle(V1_Vector, V2_Vector)) # angle between the vectors added to the global list
    
    # same process is applied to ither signals
    # angle 1: (t_body_acc_jerk[X.mean,Y.mean,Z.mean],t_gravity[X.mean,Y.mean,Z.mean]
    V1_columns=['t_body_acc_jerk_X','t_body_acc_jerk_Y','t_body_acc_jerk_Z']
    V1_Vector=np.array(t_window[V1_columns].mean())
    angles_list.append(angle(V1_Vector, V2_Vector))
    
    # angle 2: (t_body_gyro[X.mean,Y.mean,Z.mean],t_gravity[X.mean,Y.mean,Z.mean]
    V1_columns=['t_body_gyro_X','t_body_gyro_Y','t_body_gyro_Z']
    V1_Vector=np.array(t_window[V1_columns].mean())
    angles_list.append(angle(V1_Vector, V2_Vector))
    
    # angle 3: (t_body_gyro_jerk[X.mean,Y.mean,Z.mean],t_gravity[X.mean,Y.mean,Z.mean]
    V1_columns=['t_body_gyro_jerk_X','t_body_gyro_jerk_Y','t_body_gyro_jerk_Z']
    V1_Vector=np.array(t_window[V1_columns].mean())
    angles_list.append(angle(V1_Vector, V2_Vector))
    #################################################################################
    
    # V1 vector in this case is the X axis itself [1,0,0]
    # angle 4: ([X_axis],t_gravity[X.mean,Y.mean,Z.mean])   
    V1_Vector=np.array([1,0,0])
    angles_list.append(angle(V1_Vector, V2_Vector))
    
    # V1 vector in this case is the Y axis itself [0,1,0]
    # angle 5: ([Y_acc_axis],t_gravity[X.mean,Y.mean,Z.mean]) 
    V1_Vector=np.array([0,1,0])
    angles_list.append(angle(V1_Vector, V2_Vector))
    
    # V1 vector in this case is the Z axis itself [0,0,1]
    # angle 6: ([Z_acc_axis],t_gravity[X.mean,Y.mean,Z.mean])
    V1_Vector=np.array([0,0,1])
    angles_list.append(angle(V1_Vector, V2_Vector))
    
    return angles_list