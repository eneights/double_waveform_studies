import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import norm


# Reads csv file with header and time & voltage columns
# Returns time array, voltage array, and header as a string
def rw(file_name, nhdr):
    header = []
    header_str = ''
    x = np.array([])
    y = np.array([])

    if os.path.isfile(file_name):
        myfile = open(file_name, 'rb')          # Opens waveform file
        for i in range(nhdr):                   # Reads header and saves in a list
            header.append(myfile.readline())
        for line in myfile:
            x = np.append(x, float(line.split(str.encode(','))[0]))     # Reads time values & saves in an array
            y = np.append(y, float(line.split(str.encode(','))[1]))     # Reads voltage values & saves in an array
        myfile.close()                          # Closes waveform file
        head_len = len(header)
        for i in range(0, head_len):            # Converts header list to a string
            head_byte = header[i]
            head_str = head_byte.decode('cp437')
            header_str += head_str

    return x, y, header_str


# Given a time array, voltage array, and header, writes a csv file with header and time & voltage columns
def ww(x, y, file_name, hdr):
    myfile = open(file_name, 'w')           # Opens file to write waveform into
    for entry in str(hdr):                  # Writes header to file
        myfile.write(entry)
    for ix, iy in zip(x, y):                # Writes time and voltage values into file
        line = '%.7E,%f\n' % (ix, iy)
        myfile.write(line)
    myfile.close()                          # Closes waveform file


# Returns the average baseline (baseline noise level)
def calculate_average(t, v):
    v_sum = 0

    idx = np.where(v == min(v))     # Finds index of point of minimum voltage value

    if idx > len(t) / 2:            # If minimum voltage is in second half of voltage array, calculates baseline using
        idx1 = int(.1 * len(t))     # first half of voltage array
        idx2 = int(.35 * len(t))
    else:
        idx1 = int(.65 * len(t))    # If minimum voltage is in first half of voltage array, calculates baseline using
        idx2 = int(.9 * len(t))     # second half of voltage array
    for i in range(idx1, idx2):
        v_sum += v[i]
    average = v_sum / (idx2 - idx1)

    return average


# Shifts spes so that baseline = 0 and when t = 0, v = 50% max
def shift_waveform(file_num, nhdr, data_path, save_path):
    file_name = 'D1--waveforms--%05d.txt' % file_num

    if os.path.isfile(data_path / file_name):
        if os.path.isfile(save_path / file_name):       # If file has already been shifted, does nothing
            pass
        else:
            t, v, hdr = rw(data_path / file_name, nhdr)     # Reads waveform file
            half_max = min(v) / 2                           # Calculates 50% max
            differential = np.diff(v)                       # Calculates derivative of every point in voltage array
            difference_value = np.abs(v - half_max)   # Finds difference between every point in voltage array & 50% max
            for i in range(0, len(differential)):       # Sets every value in difference_value array with a positive
                if differential[i] > 0:                 # derivative equal to infinity
                    difference_value[i] = np.inf
            index = np.argmin(difference_value)  # Finds index of closest voltage to 50% max with a negative derivative
            half_max_time = t[index]            # Finds time at 50% max
            t2 = t - half_max_time              # Subtracts time of 50% max from time array
            avg = calculate_average(t, v)       # Calculates average baseline
            v2 = v - avg                        # Subtracts average baseline voltage from voltage array
            ww(t2, v2, save_path / file_name, hdr)      # Writes shifted waveform to file
            print('Length of /d1_shifted/:', len(os.listdir(str(save_path))))


# Returns charge of spe (as a positive value)
def calculate_charge(t, v, r):
    vsum = 0
    tvals = np.linspace(t[0], t[len(t) - 1], 5000)      # Creates array of times over entire timespan
    vvals = np.interp(tvals, t, v)                      # Interpolates & creates array of voltages over entire timespan

    for i in range(len(tvals)):                         # Calculates sum of all voltages in full timespan
        vsum += vvals[i]
    charge = -1 * (tvals[len(tvals) - 1]) * vsum / (len(tvals) * r)     # Calculates charge

    return charge


# Returns time when spe waveform begins and time when spe waveform ends
def calculate_t1_t2(t, v):
    idx1 = np.inf
    idx2 = np.inf

    min_time = t[np.where(v == min(v))][0]              # Finds time of point of minimum voltage

    tvals = np.linspace(t[0], t[len(t) - 1], 5000)      # Creates array of times over entire timespan
    tvals1 = np.linspace(t[0], min_time, 5000)          # Creates array of times from beginning to point of min voltage
    tvals2 = np.linspace(min_time, t[len(t) - 1], 5000)     # Creates array of times from point of min voltage to end
    vvals1 = np.interp(tvals1, t, v)   # Interpolates & creates array of voltages from beginning to point of min voltage
    vvals2 = np.interp(tvals2, t, v)    # Interpolates & creates array of voltages from point of min voltage to end
    vvals1_flip = np.flip(vvals1)   # Flips array, creating array of voltages from point of min voltage to beginning
    difference_value1 = vvals1_flip - (0.1 * min(v))    # Finds difference between points in beginning array and 10% max
    difference_value2 = vvals2 - (0.1 * min(v))         # Finds difference between points in end array and 10% max

    for i in range(0, len(difference_value1) - 1):  # Starting at point of minimum voltage and going towards beginning
        if difference_value1[i] >= 0:               # of waveform, finds where voltage becomes greater than 10% max
            idx1 = len(difference_value1) - i
            break
    if idx1 == np.inf:      # If voltage never becomes greater than 10% max, finds where voltage is closest to 10% max
        idx1 = len(difference_value1) - 1 - np.argmin(np.abs(difference_value1))
    for i in range(0, len(difference_value2) - 1):      # Starting at point of minimum voltage and going towards end of
        if difference_value2[i] >= 0:                   # waveform, finds where voltage becomes greater than 10% max
            idx2 = i
            break
    if idx2 == np.inf:      # If voltage never becomes greater than 10% max, finds where voltage is closest to 10% max
        idx2 = np.argmin(np.abs(difference_value2))

    t1 = tvals[np.argmin(np.abs(tvals - tvals1[idx1]))]             # Finds time of beginning of spe
    t2 = tvals[np.argmin(np.abs(tvals - tvals2[idx2]))]             # Finds time of end of spe

    return t1, t2


# Returns the amplitude of spe as a positive value (minimum voltage)
def calculate_amp(t, v):
    avg = calculate_average(t, v)       # Calculates value of baseline voltage
    amp = avg - np.amin(v)              # Calculates max amplitude

    return amp


# Returns the full width half max (FWHM) of spe
def calculate_fwhm(t, v):
    half_max = (min(v) / 2).item()                      # Calculates 50% max value
    tvals = np.linspace(t[0], t[len(t) - 1], 5000)      # Creates array of times over entire timespan
    vvals = np.interp(tvals, t, v)                      # Interpolates & creates array of voltages over entire timespan
    difference_value = np.abs(vvals - half_max)         # Finds difference between points in voltage array and 50% max
    index_min = np.argmin(np.abs(vvals - min(v)))       # Finds index of minimum voltage in voltage array
    for i in range(index_min.item(), len(np.diff(vvals)) - 1):  # Sets every value in difference_value array with a
        if np.diff(vvals)[i] < 0:                               # negative differential equal to infinity
            difference_value[i] = np.inf
    difference_value = difference_value[index_min.item():len(vvals) - 1]
    idx = np.argmin(difference_value)       # Finds index of 50% max in voltage array
    half_max_time = tvals[idx + index_min.item()]   # Finds time of 50% max

    return half_max_time
