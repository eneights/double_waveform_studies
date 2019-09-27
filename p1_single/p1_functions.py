import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import norm

# FILE READING/WRITING


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


# Creates text file with time of beginning of spe, time of end of spe, charge, amplitude, fwhm, 10-90 & 20-80 rise
# times, 10-90 & 20-80 fall times, and 10%, 20%, 80% & 90% jitter for an spe file
def save_calculations(dest_path, i, t1, t2, charge, amplitude, fwhm, rise1090, rise2080, fall1090, fall2080, time10,
                      time20, time80, time90):
    file_name = str(dest_path / 'calculations' / 'D1--waveforms--%05d.txt') % i
    myfile = open(file_name, 'w')
    myfile.write('t1,' + str(t1))
    myfile.write('\nt2,' + str(t2))
    myfile.write('\ncharge,' + str(charge))
    myfile.write('\namplitude,' + str(amplitude))
    myfile.write('\nfwhm,' + str(fwhm))
    myfile.write('\nrise1090,' + str(rise1090))
    myfile.write('\nrise2080,' + str(rise2080))
    myfile.write('\nfall1090,' + str(fall1090))
    myfile.write('\nfall2080,' + str(fall2080))
    myfile.write('\ntime10,' + str(time10))
    myfile.write('\ntime20,' + str(time20))
    myfile.write('\ntime80,' + str(time80))
    myfile.write('\ntime90,' + str(time90))
    myfile.close()


# Creates text file with data from an array
def write_hist_data(array, dest_path, name):
    array = np.sort(array)
    file_name = Path(dest_path / 'hist_data' / name)

    myfile = open(file_name, 'w')
    for item in array:  # Writes an array item on each line of file
        myfile.write(str(item) + '\n')
    myfile.close()


# Checks if a file exists
def check_if_file(path, file_name):
    if os.path.isfile(path / file_name):
        return 'yes'
    else:
        return 'no'


# Reads calculations from existing file
def read_calc(filename):
    myfile = open(filename, 'r')  # Opens file with calculations
    csv_reader = csv.reader(myfile)
    file_array = np.array([])
    for row in csv_reader:  # Creates array with calculation data
        file_array = np.append(file_array, float(row[1]))
    myfile.close()
    t1 = file_array[0]
    t2 = file_array[1]
    charge = file_array[2]
    amp = file_array[3]
    fwhm = file_array[4]
    rise1090 = file_array[5]
    rise2080 = file_array[6]
    fall1090 = file_array[7]
    fall2080 = file_array[8]
    j10 = file_array[9]
    j20 = file_array[10]
    j80 = file_array[11]
    j90 = file_array[12]

    return t1, t2, charge, amp, fwhm, rise1090, rise2080, fall1090, fall2080, j10, j20, j80, j90


# CALCULATIONS


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
    half_max_time = tvals[np.argmin(difference_value) + index_min.item()]   # Finds time of 50% max

    return half_max_time


# Returns rise times of given percentages of amplitude
def rise_time(t, v, low, high):
    percent_low = low / 100
    percent_high = high / 100

    avg = calculate_average(t, v)               # Calculates average baseline
    t1, t2 = calculate_t1_t2(t, v)              # Calculates start time of spe
    min_time = t[np.where(v == min(v))][0]      # Finds time at point of minimum voltage

    val_1 = percent_low * (min(v) - avg)        # Calculates first percent of max
    val_2 = percent_high * (min(v) - avg)       # Calculates second percent of max

    tvals = np.linspace(t1, min_time, 5000) # Creates array of times from beginning of spe to point of minimum voltage
    vvals = np.interp(tvals, t, v)  # Interpolates & creates array of voltages from beginning of spe to minimum voltage

    time_low = tvals[np.argmin(np.abs(vvals - val_1))]          # Finds time of point of first percent of max
    time_high = tvals[np.argmin(np.abs(vvals - val_2))]         # Finds time of point of second percent of max

    risetime = time_high - time_low                             # Calculates rise time
    risetime = float(format(risetime, '.2e'))

    return risetime


# Returns fall times of given percentages of amplitude
def fall_time(t, v, low, high):
    percent_low = low / 100
    percent_high = high / 100

    avg = calculate_average(t, v)               # Calculates average baseline
    t1, t2 = calculate_t1_t2(t, v)              # Calculates start time of spe
    min_time = t[np.where(v == min(v))][0]      # Finds time at point of minimum voltage

    val_1 = percent_high * (min(v) - avg)       # Calculates first percent of max
    val_2 = percent_low * (min(v) - avg)        # Calculates second percent of max

    tvals = np.linspace(min_time, t2, 5000)  # Creates array of times from beginning of spe to point of minimum voltage
    vvals = np.interp(tvals, t, v)  # Interpolates & creates array of voltages from beginning of spe to minimum voltage

    time_high = tvals[np.argmin(np.abs(vvals - val_1))]     # Finds time of point of first percent of max
    time_low = tvals[np.argmin(np.abs(vvals - val_2))]      # Finds time of point of second percent of max

    falltime = time_low - time_high                         # Calculates fall time
    falltime = float(format(falltime, '.2e'))

    return falltime


# Returns percent jitter of a given percent
def calculate_jitter(t, v, per):
    percent = per / 100

    avg = calculate_average(t, v)               # Calculates average baseline
    t1, t2 = calculate_t1_t2(t, v)              # Calculates start time of spe
    min_time = t[np.where(v == min(v))][0]      # Finds time at point of minimum voltage

    val = percent * (min(v) - avg)              # Calculates percent of max
    tvals = np.linspace(t1, min_time, 5000) # Creates array of times from beginning of spe to point of minimum voltage
    vvals = np.interp(tvals, t, v)  # Interpolates & creates array of voltages from beginning of spe to minimum voltage

    time = tvals[np.argmin(np.abs(vvals - val))]        # Finds time

    return time


# DOING CALCULATIONS


# Checks if calculated values are possible or not
def check_if_impossible(t1, t2, charge, amp, fwhm, rise1090, rise2080, fall1090, fall2080, j10, j20, j80, j90):
    if (t1 < 0 or t2 <= t1 or charge <= 0 or amp <= 0 or fwhm <= 0 or rise1090 <= 0 or rise2080 <= 0 or fall1090 <= 0 or
            fall2080 <= 0 or j10 >= 0 or j20 >= 0 or j80 <= 0 or j90 <= 0):
        return 'impossible'
    else:
        return 'ok'


# Creates array for a calculation
def make_array():
    my_array = np.array([])



    return my_array


# Removes spe waveform from all spe folders
def remove_spe(path_1, path_2, number, nhdr):
    t, v, hdr = rw(str(path_1 / 'C2--waveforms--%05d.txt') % number, nhdr)
    ww(t, v, str(path_2 / 'not_spe' / 'D1--not_spe--%05d.txt') % number, hdr)
    if 

    os.remove(str(save_shift / 'D1--waveforms--%05d.txt') % i)
    os.remove(str(dest_path / 'd1_raw' / 'D1--waveforms--%05d.txt') % i)
    os.remove(str(dest_path / 'calculations' / 'D1--waveforms--%05d.txt') % i)




# Calculates beginning & end times of spe waveform, charge, amplitude, fwhm, 10-90 & 20-80 rise times, 10-90 & 20-80
# fall times, and 10%, 20%, 80% & 90% jitter for each spe file
# Returns arrays of beginning & end times of spe waveform, charge, amplitude, fwhm, 10-90 & 20-80 rise times, 10-90 &
# 20-80 fall times, and 10%, 20%, 80% & 90% jitter
def make_arrays(save_shift, dest_path, data_sort, start, end, nhdr, r):
    t1_array = np.array([])
    t2_array = np.array([])
    charge_array = np.array([])
    amplitude_array = np.array([])
    fwhm_array = np.array([])
    rise1090_array = np.array([])
    rise2080_array = np.array([])
    fall1090_array = np.array([])
    fall2080_array = np.array([])
    time10_array = np.array([])
    time20_array = np.array([])
    time80_array = np.array([])
    time90_array = np.array([])

    for i in range(start, end + 1):
        file_name1 = str(save_shift / 'D1--waveforms--%05d.txt') % i
        file_name2 = str(dest_path / 'calculations' / 'D1--waveforms--%05d.txt') % i
        if os.path.isfile(file_name1):
            if os.path.isfile(file_name2):      # If the calculations were done previously, they are read from a file
                t1, t2, charge, amplitude, fwhm, rise1090, rise2080, fall1090, fall2080, time10, time20, time80, time90\
                    = read_calc(file_name2)
                possibility = check_if_impossible(t1, t2, charge, amplitude, fwhm, rise1090, rise2080, fall1090,
                                                  fall2080, time10, time20, time80, time90)
                # Any spe waveform that returns impossible values is put into the not_spe folder
                if possibility == 'impossible':
                    raw_file = str(data_sort / 'C2--waveforms--%05d.txt') % i
                    save_file = str(dest_path / 'not_spe' / 'D1--not_spe--%05d.txt') % i
                    t, v, hdr = rw(raw_file, nhdr)
                    ww(t, v, save_file, hdr)
                    print('Removing file #%05d' % i)
                    os.remove(str(save_shift / 'D1--waveforms--%05d.txt') % i)
                    os.remove(str(dest_path / 'd1_raw' / 'D1--waveforms--%05d.txt') % i)
                    os.remove(str(dest_path / 'calculations' / 'D1--waveforms--%05d.txt') % i)
                # All other spe waveforms' calculations are placed into arrays
                else:
                    t1_array = np.append(t1_array, t1)
                    t2_array = np.append(t2_array, t2)
                    charge_array = np.append(charge_array, charge)
                    amplitude_array = np.append(amplitude_array, amplitude)
                    fwhm_array = np.append(fwhm_array, fwhm)
                    rise1090_array = np.append(rise1090_array, rise1090)
                    rise2080_array = np.append(rise2080_array, rise2080)
                    fall1090_array = np.append(fall1090_array, fall1090)
                    fall2080_array = np.append(fall2080_array, fall2080)
                    time10_array = np.append(time10_array, time10)
                    time20_array = np.append(time20_array, time20)
                    time80_array = np.append(time80_array, time80)
                    time90_array = np.append(time90_array, time90)
            else:           # If the calculations were not done yet, they are calculated
                print("Calculating shifted file #%05d" % i)
                t, v, hdr = rw(file_name1, nhdr)        # Shifted waveform file is read
                t1, t2, charge = calculate_charge(t, v, r)      # Start & end times and charge of spe are calculated
                amplitude = calculate_amp(t, v)     # Amplitude of spe is calculated
                fwhm = calculate_fwhm(t, v)         # FWHM of spe is calculated
                rise1090, rise2080 = rise_time(t, v, r)     # 10-90 & 20-80 rise times of spe are calculated
                fall1090, fall2080 = fall_time(t, v, r)     # 10-90 & 20-80 fall times of spe are calculated
                time10, time20, time80, time90 = calculate_times(t, v, r)   # 10%, 20%, 80% & 90% jitter is calculated
                possibility = check_if_impossible(t1, t2, charge, amplitude, fwhm, rise1090, rise2080, fall1090,
                                                  fall2080, time10, time20, time80, time90)
                # Any spe waveform that returns impossible values is put into the not_spe folder
                if possibility == 'impossible':
                    raw_file = str(data_sort / 'C2--waveforms--%05d.txt') % i
                    save_file = str(dest_path / 'not_spe' / 'D1--not_spe--%05d.txt') % i
                    t, v, hdr = rw(raw_file, nhdr)
                    ww(t, v, save_file, hdr)
                    print('Removing file #%05d' % i)
                    os.remove(str(save_shift / 'D1--waveforms--%05d.txt') % i)
                    os.remove(str(dest_path / 'd1_raw' / 'D1--waveforms--%05d.txt') % i)
                # All other spe waveforms' calculations are saved in a file & placed into arrays
                else:
                    save_calculations(dest_path, i, t1, t2, charge, amplitude, fwhm, rise1090, rise2080, fall1090,
                                      fall2080, time10, time20, time80, time90)
                    t1_array = np.append(t1_array, t1)
                    t2_array = np.append(t2_array, t2)
                    charge_array = np.append(charge_array, charge)
                    amplitude_array = np.append(amplitude_array, amplitude)
                    fwhm_array = np.append(fwhm_array, fwhm)
                    rise1090_array = np.append(rise1090_array, rise1090)
                    rise2080_array = np.append(rise2080_array, rise2080)
                    fall1090_array = np.append(fall1090_array, fall1090)
                    fall2080_array = np.append(fall2080_array, fall2080)
                    time10_array = np.append(time10_array, time10)
                    time20_array = np.append(time20_array, time20)
                    time80_array = np.append(time80_array, time80)
                    time90_array = np.append(time90_array, time90)

    return t1_array, t2_array, charge_array, amplitude_array, fwhm_array, rise1090_array, rise2080_array, \
        fall1090_array, fall2080_array, time10_array, time20_array, time80_array, time90_array


