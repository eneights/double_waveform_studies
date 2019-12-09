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
def average_waveform(start, end, dest_path, shaping, shaping_name, nhdr, fsps_new):
    data_file = Path(Path(dest_path / (str(shaping + '_single')) / str('digitized_' + str(int(fsps_new / 1e6)) +
                                                                       '_Msps')))
    save_file = Path(dest_path / 'plots')
    tsum = 0
    vsum = 0
    n = 0

    for i in range(start, end + 1):
        file_name = 'D3--waveforms--%05d.txt' % i
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
    plt.title('Average Waveform (' + shaping_name + ', ' + str(int(fsps_new / 1e6)) + ' Msps' + ')')
    plt.savefig(save_file / str('avg_waveform_single_' + str(int(fsps_new / 1e6)) + ' Msps_' + shaping + '.png'),
                dpi=360)

    # Saves average waveform data
    file_name = dest_path / 'hist_data_single' / str('avg_waveform_' + str(int(fsps_new / 1e6)) + ' Msps_' + shaping +
                                                     '.txt')
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
def make_folders(dest_path, filt_path1, filt_path2, filt_path4, filt_path8, fsps_new):
    if not os.path.exists(dest_path):
        print('Creating d3 folder')
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
    if not os.path.exists(Path(filt_path1 / 'raw')):
        print('Creating rt 1 raw folder')
        os.mkdir(Path(filt_path1 / 'raw'))
    if not os.path.exists(Path(filt_path2 / 'raw')):
        print('Creating rt 2 raw folder')
        os.mkdir(Path(filt_path2 / 'raw'))
    if not os.path.exists(Path(filt_path4 / 'raw')):
        print('Creating rt 4 raw folder')
        os.mkdir(Path(filt_path4 / 'raw'))
    if not os.path.exists(Path(filt_path8 / 'raw')):
        print('Creating rt 8 raw folder')
        os.mkdir(Path(filt_path8 / 'raw'))
    if not os.path.exists(Path(filt_path1 / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps'))):
        print('Creating rt 1 downsampled folder')
        os.mkdir(Path(filt_path1 / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps')))
    if not os.path.exists(Path(filt_path2 / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps'))):
        print('Creating rt 2 downsampled folder')
        os.mkdir(Path(filt_path2 / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps')))
    if not os.path.exists(Path(filt_path4 / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps'))):
        print('Creating rt 4 downsampled folder')
        os.mkdir(Path(filt_path4 / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps')))
    if not os.path.exists(Path(filt_path8 / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps'))):
        print('Creating rt 8 downsampled folder')
        os.mkdir(Path(filt_path8 / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps')))
    if not os.path.exists(Path(filt_path1 / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps'))):
        print('Creating rt 1 digitized folder')
        os.mkdir(Path(filt_path1 / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps')))
    if not os.path.exists(Path(filt_path2 / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps'))):
        print('Creating rt 2 digitized folder')
        os.mkdir(Path(filt_path2 / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps')))
    if not os.path.exists(Path(filt_path4 / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps'))):
        print('Creating rt 4 digitized folder')
        os.mkdir(Path(filt_path4 / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps')))
    if not os.path.exists(Path(filt_path8 / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps'))):
        print('Creating rt 8 digitized folder')
        os.mkdir(Path(filt_path8 / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps')))
    if not os.path.exists(Path(dest_path / 'hist_data_single')):
        print('Creating histogram data folder')
        os.mkdir(Path(dest_path / 'hist_data_single'))
    if not os.path.exists(Path(dest_path / 'plots')):
        print('Creating plots folder')
        os.mkdir(Path(dest_path / 'plots'))
    if not os.path.exists(Path(dest_path / 'unusable_data')):
        print('Creating unusable data folder')
        os.mkdir(Path(dest_path / 'unusable_data'))


# Transfer initial files to d3 folders
def transfer_files(data_path, filt_path1, filt_path2, filt_path4, filt_path8, i, nhdr):
    file_name1 = str(data_path / 'rt_1_single' / 'D2--waveforms--%05d.txt') % i
    file_name2 = str(data_path / 'rt_2_single' / 'D2--waveforms--%05d.txt') % i
    file_name4 = str(data_path / 'rt_4_single' / 'D2--waveforms--%05d.txt') % i
    file_name8 = str(data_path / 'rt_8_single' / 'D2--waveforms--%05d.txt') % i
    save_name1 = str(filt_path1 / 'raw' / 'D3--waveforms--%05d.txt') % i
    save_name2 = str(filt_path2 / 'raw' / 'D3--waveforms--%05d.txt') % i
    save_name4 = str(filt_path4 / 'raw' / 'D3--waveforms--%05d.txt') % i
    save_name8 = str(filt_path8 / 'raw' / 'D3--waveforms--%05d.txt') % i

    if os.path.isfile(file_name1):
        if os.path.isfile(save_name1):
            print('File #%05d in rt_1 folder' % i)
        else:
            t, v, hdr = rw(file_name1, nhdr)
            ww(t, v, save_name1, hdr)
            print('File #%05d in rt_1 folder' % i)

    if os.path.isfile(file_name2):
        if os.path.isfile(save_name2):
            print('File #%05d in rt_2 folder' % i)
        else:
            t, v, hdr = rw(file_name2, nhdr)
            ww(t, v, save_name2, hdr)
            print('File #%05d in rt_2 folder' % i)

    if os.path.isfile(file_name4):
        if os.path.isfile(save_name4):
            print('File #%05d in rt_4 folder' % i)
        else:
            t, v, hdr = rw(file_name4, nhdr)
            ww(t, v, save_name4, hdr)
            print('File #%05d in rt_4 folder' % i)

    if os.path.isfile(file_name8):
        if os.path.isfile(save_name8):
            print('File #%05d in rt_8 folder' % i)
        else:
            t, v, hdr = rw(file_name8, nhdr)
            ww(t, v, save_name8, hdr)
            print('File #%05d in rt_8 folder' % i)


# Given a time array, voltage array, sample rate, and new sample rate, creates downsampled time and voltage arrays
def downsample(t, v, fsps, fsps_new):
    steps = int(fsps / fsps_new + 0.5)
    idx_start = random.randint(0, steps - 1)        # Picks a random index to start at
    t_ds = np.array([])
    v_ds = np.array([])
    for i in range(idx_start, len(v) - 1, steps):   # Creates time & voltage arrays that digitizer would detect
        t_ds = np.append(t_ds, t[i])
        v_ds = np.append(v_ds, v[i])
    return t_ds, v_ds


# Converts voltage array to bits and adds noise
def digitize(v, noise):
    v_bits = np.array([])
    for i in range(len(v)):
        v_bits = np.append(v_bits, (v[i] * (2 ** 14 - 1) * 2 + 0.5))        # Converts voltage array to bits
    noise_array = np.random.normal(scale=noise, size=len(v_bits))           # Creates noise array
    v_digitized = np.add(v_bits, noise_array)                               # Adds noise to digitized values
    v_digitized = v_digitized.astype(int)
    return v_digitized


# Downsamples and digitizes files
def down_dig(filt_path1, filt_path2, filt_path4, filt_path8, fsps, fsps_new, noise, start, end, nhdr):
    for i in range(start, end + 1):
        file_name1 = str(filt_path1 / 'raw' / 'D3--waveforms--%05d.txt') % i
        file_name2 = str(filt_path2 / 'raw' / 'D3--waveforms--%05d.txt') % i
        file_name4 = str(filt_path4 / 'raw' / 'D3--waveforms--%05d.txt') % i
        file_name8 = str(filt_path8 / 'raw' / 'D3--waveforms--%05d.txt') % i
        down_name1 = str(filt_path1 / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') /
                         'D3--waveforms--%05d.txt') % i
        down_name2 = str(filt_path2 / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') /
                         'D3--waveforms--%05d.txt') % i
        down_name4 = str(filt_path4 / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') /
                         'D3--waveforms--%05d.txt') % i
        down_name8 = str(filt_path8 / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') /
                         'D3--waveforms--%05d.txt') % i
        dig_name1 = str(filt_path1 / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') /
                        'D3--waveforms--%05d.txt') % i
        dig_name2 = str(filt_path2 / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') /
                        'D3--waveforms--%05d.txt') % i
        dig_name4 = str(filt_path4 / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') /
                        'D3--waveforms--%05d.txt') % i
        dig_name8 = str(filt_path8 / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') /
                        'D3--waveforms--%05d.txt') % i

        if os.path.isfile(down_name1) and os.path.isfile(down_name2) and os.path.isfile(down_name4) and \
                os.path.isfile(down_name8):
            print('File #%05d downsampled' % i)
        else:
            if os.path.isfile(file_name1) and os.path.isfile(file_name2) and os.path.isfile(file_name4) and \
                    os.path.isfile(file_name8):
                print('Downsampling file #%05d' % i)
                if not os.path.isfile(down_name1):
                    t, v, hdr = rw(file_name1, nhdr)
                    t_ds, v_ds = downsample(t, v, fsps, fsps_new)
                    ww(t_ds, v_ds, down_name1, hdr)
                if not os.path.isfile(down_name2):
                    t, v, hdr = rw(file_name2, nhdr)
                    t_ds, v_ds = downsample(t, v, fsps, fsps_new)
                    ww(t_ds, v_ds, down_name2, hdr)
                if not os.path.isfile(down_name4):
                    t, v, hdr = rw(file_name4, nhdr)
                    t_ds, v_ds = downsample(t, v, fsps, fsps_new)
                    ww(t_ds, v_ds, down_name4, hdr)
                if not os.path.isfile(down_name8):
                    t, v, hdr = rw(file_name8, nhdr)
                    t_ds, v_ds = downsample(t, v, fsps, fsps_new)
                    ww(t_ds, v_ds, down_name8, hdr)

        if os.path.isfile(dig_name1) and os.path.isfile(dig_name2) and os.path.isfile(dig_name4) and \
                os.path.isfile(dig_name8):
            print('File #%05d digitized' % i)
        else:
            if os.path.isfile(file_name1) and os.path.isfile(file_name2) and os.path.isfile(file_name4) and \
                    os.path.isfile(file_name8):
                print('Digitizing file #%05d' % i)
                if not os.path.isfile(dig_name1):
                    t, v, hdr = rw(file_name1, nhdr)
                    v_dig = digitize(v, noise)
                    ww(t, v_dig, dig_name1, hdr)
                if not os.path.isfile(dig_name2):
                    t, v, hdr = rw(file_name2, nhdr)
                    v_dig = digitize(v, noise)
                    ww(t, v_dig, dig_name2, hdr)
                if not os.path.isfile(dig_name4):
                    t, v, hdr = rw(file_name4, nhdr)
                    v_dig = digitize(v, noise)
                    ww(t, v_dig, dig_name4, hdr)
                if not os.path.isfile(dig_name8):
                    t, v, hdr = rw(file_name8, nhdr)
                    v_dig = digitize(v, noise)
                    ww(t, v_dig, dig_name8, hdr)
