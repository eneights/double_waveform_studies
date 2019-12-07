import os
import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy import signal
import random

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


# Creates text file with rise times at each shaping
def save_calculations(dest_path, i, risetime_1, risetime_2, risetime_4, risetime_8):
    file_name = str(dest_path / 'calculations_single' / 'D2--waveforms--%05d.txt') % i
    myfile = open(file_name, 'w')
    myfile.write('risetime_1,' + str(risetime_1))
    myfile.write('\nrisetime_2,' + str(risetime_2))
    myfile.write('\nrisetime_4,' + str(risetime_4))
    myfile.write('\nrisetime_8,' + str(risetime_8))
    myfile.close()


# Reads calculation file
def read_calc(filename):
    myfile = open(filename, 'r')  # Opens file with calculations
    csv_reader = csv.reader(myfile)
    file_array = np.array([])
    for row in csv_reader:  # Creates array with calculation data
        file_array = np.append(file_array, float(row[1]))
    myfile.close()
    risetime_1 = file_array[0]
    risetime_2 = file_array[1]
    risetime_4 = file_array[2]
    risetime_8 = file_array[3]

    return risetime_1, risetime_2, risetime_4, risetime_8


# Creates text file with data from an array
def write_hist_data(array, dest_path, name):
    array = np.sort(array)
    file_name = Path(dest_path / 'hist_data_single' / name)

    myfile = open(file_name, 'w')
    for item in array:  # Writes an array item on each line of file
        myfile.write(str(item) + '\n')
    myfile.close()


# Creates info file
def info_file(acq_date_time, source_path, dest_path, pmt_hv, gain, offset, trig_delay, amp, fsps, band, nfilter, r):
    now = datetime.datetime.now()
    file_name = 'info.txt'
    file = dest_path / file_name
    myfile = open(file, 'w')
    myfile.write('Data acquisition,' + str(acq_date_time))              # date & time of raw data from d0 info file
    myfile.write('\nData processing,' + str(now))                       # current date & time
    myfile.write('\nSource data,' + str(source_path))                   # path to source data
    myfile.write('\nDestination data,' + str(dest_path))                # path to folder of current data
    myfile.write('\nPMT HV (V),' + str(pmt_hv))                         # voltage of PMT from d0 info file
    myfile.write('\nNominal gain,' + str(gain))                         # gain of PMT from d0 info file
    myfile.write('\nDG 535 offset,' + str(offset))                      # offset of pulse generator from d0 info file
    myfile.write('\nDG 535 trigger delay (ns),' + str(trig_delay))      # trigger delay of pulse generator from d0 info
                                                                        # file
    myfile.write('\nDG 535 amplitude (V),' + str(amp))                  # amplitude of pulse generator from d0 info file
    myfile.write('\nOscilloscope sample rate (Hz),' + str(fsps))        # sample rate of oscilloscope from d0 info file
    myfile.write('\nOscilloscope bandwidth (Hz),' + str(band))          # bandwidth of oscilloscope from d0 info file
    myfile.write('\nOscilloscope noise filter (bits),' + str(nfilter))  # oscilloscope noise filter from d0 info file
    myfile.write('\nOscilloscope resistance (ohms),' + str(r))          # resistance of oscilloscope from d0 info file
    myfile.close()


# Reads info file
def read_info(myfile):
    csv_reader = csv.reader(myfile)
    info_array = np.array([])
    for row in csv_reader:
        info_array = np.append(info_array, row[1])
    i_date_time = info_array[0]
    i_path = info_array[1]
    i_nhdr = int(info_array[2])
    i_baseline = float(info_array[3])
    i_pmt_hv = int(info_array[4])
    i_gain = int(float(info_array[5]))
    i_offset = int(info_array[6])
    i_trig_delay = float(info_array[7])
    i_amp = float(info_array[8])
    i_fsps = float(info_array[9])
    i_band = info_array[10]
    i_nfilter = float(info_array[11])
    i_r = int(info_array[12])
    a, b, c, d, e, fol, f, i_fil_band, g = i_path.split('/')
    i_date, watch, spe = fol.split('_')
    i_date = int(i_date)

    return i_date, i_date_time, i_fil_band, i_nhdr, i_fsps, i_baseline, i_r, i_pmt_hv, i_gain, i_offset, i_trig_delay, \
        i_amp, i_band, i_nfilter


# AVERAGE/PLOT WAVEFORM


# Calculates average waveform of spe
def average_waveform(start, end, dest_path, shaping, shaping_name, nhdr):
    data_file = Path(dest_path / str(shaping + '_single'))
    save_file = Path(dest_path / 'plots')
    tsum = 0
    vsum = 0
    n = 0

    for i in range(start, end + 1):
        file_name = 'D2--waveforms--%05d.txt' % i
        if os.path.isfile(data_file / file_name):
            print('Reading file #', i)
            t, v, hdr = rw(data_file / file_name, nhdr)     # Reads a waveform file
            array_length = len(t)
            v = v / min(v)                                  # Normalizes voltages
            idx = np.where(t == 0)                          # Finds index of t = 0 point
            idx = int(idx[0])
            t = np.roll(t, -idx)                            # Rolls time array so that t = 0 point is at index 0
            v = np.roll(v, -idx)                            # Rolls voltage array so that 50% max point is at index 0
            idx2 = np.where(t == min(t))                    # Finds index of point of minimum t
            idx2 = int(idx2[0])
            idx3 = np.where(t == max(t))                    # Finds index of point of maximum t
            idx3 = int(idx3[0])
            # Only averages waveform files that have enough points before t = 0 & after the spe
            if idx2 <= int(0.87 * array_length):
                # Removes points between point of maximum t & chosen minimum t in time & voltage arrays
                t = np.concatenate((t[:idx3], t[int(0.87 * array_length):]))
                v = np.concatenate((v[:idx3], v[int(0.87 * array_length):]))
                # Rolls time & voltage arrays so that point of chosen minimum t is at index 0
                t = np.roll(t, -idx3)
                v = np.roll(v, -idx3)
                if len(t) >= int(0.99 * array_length):
                    # Removes points after chosen point of maximum t in time & voltage arrays
                    t = t[:int(0.99 * array_length)]
                    v = v[:int(0.99 * array_length)]
                    # Sums time & voltage arrays
                    tsum += t
                    vsum += v
                    n += 1
    # Finds average time & voltage arrays
    t_avg = tsum / n
    v_avg = vsum / n

    # Plots average waveform & saves image
    plt.plot(t_avg, v_avg)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Voltage')
    plt.title('Average Waveform (' + shaping_name + ')')
    plt.savefig(save_file / str('avg_waveform_single_' + shaping + '.png'), dpi=360)

    # Saves average waveform data
    file_name = dest_path / 'hist_data_single' / str('avg_waveform_' + shaping + '.txt')
    hdr = 'Average Waveform\n\n\n\nTime,Ampl'
    ww(t_avg, v_avg, file_name, hdr)

    plt.close()


# Shows a waveform plot to user
def show_waveform(file_name, version):
    t, v, hdr = rw(file_name, 5)
    print("\nHeader:\n\n" + str(hdr))
    plt.plot(t, v)
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.title(version + ' Waveform')
    plt.show()


# CALCULATIONS

# P3

# Creates p3 folder names
def initialize_folders(date, filter_band):
    gen_path = Path(r'/Volumes/TOSHIBA EXT/data/watchman')
    save_path = Path(str(gen_path / '%08d_watchman_spe/waveforms/%s') % (date, filter_band))
    data_path = Path(save_path / 'd2')
    dest_path = Path(save_path / 'd3')
    filt_path1 = Path(dest_path / 'rt_1_single')
    filt_path2 = Path(dest_path / 'rt_2_single')
    filt_path4 = Path(dest_path / 'rt_4_single')
    filt_path8 = Path(dest_path / 'rt_8_single')

    return gen_path, save_path, data_path, dest_path, filt_path1, filt_path2, filt_path4, filt_path8


# Creates p3 folders
def make_folders(dest_path, filt_path1, filt_path2, filt_path4, filt_path8):
    if not os.path.exists(dest_path):
        print('Creating d2 folder')
        os.mkdir(dest_path)
    if not os.path.exists(filt_path1):
        print('Creating rt 1 folder')
        os.mkdir(filt_path1)
    if not os.path.exists(filt_path2):
        print('Creating rt 2 folder')
        os.mkdir(filt_path2)
    if not os.path.exists(filt_path4):
        print('Creating rt 4 folder')
        os.mkdir(filt_path4)
    if not os.path.exists(filt_path8):
        print('Creating rt 8 folder')
        os.mkdir(filt_path8)
    if not os.path.exists(Path(dest_path / 'hist_data_single')):
        print('Creating histogram data folder')
        os.mkdir(Path(dest_path / 'hist_data_single'))
    if not os.path.exists(Path(dest_path / 'plots')):
        print('Creating plots folder')
        os.mkdir(Path(dest_path / 'plots'))
    if not os.path.exists(Path(dest_path / 'calculations_single')):
        print('Creating calculations folder')
        os.mkdir(Path(dest_path / 'calculations_single'))
    if not os.path.exists(Path(dest_path / 'unusable_data')):
        print('Creating unusable data folder')
        os.mkdir(Path(dest_path / 'unusable_data'))
