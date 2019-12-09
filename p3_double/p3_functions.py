import os
import csv
import datetime
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
def average_waveform(array, dest_path, shaping, shaping_name, delay_path, delay_name, delay_folder, nhdr, fsps_new):
    save_file = Path(dest_path / 'plots')
    tsum = 0
    vsum = 0
    n = 0

    for item in array:
        file_name = 'D3--waveforms--%s.txt' % item
        if os.path.isfile(delay_path / file_name):
            print('Reading file #', item)
            t, v, hdr = rw(delay_path / file_name, nhdr)    # Reads a waveform file
            v = v / min(v)                                  # Normalizes voltages
            try:
                idx = np.where(t == 0)                      # Finds index of t = 0 point
                idx = int(idx[0])
                t = np.roll(t, -idx)                        # Rolls time array so that t = 0 point is at index 0
                v = np.roll(v, -idx)                        # Rolls voltage array so that 50% max point is at index 0
                idx2 = int(np.where(t == min(t))[0])        # Finds index of point of minimum t
                idx3 = int(np.where(t == max(t))[0])        # Finds index of point of maximum t
                idx_low = int(np.where(t == -2.5e-8)[0])
                idx_high = int(np.where(t == 1.5e-7)[0])
                # Only averages waveform files that have enough points before t = 0 & after the spe
                if idx2 <= idx_low:
                    # Removes points between point of maximum t & chosen minimum t in time & voltage arrays
                    t = np.concatenate((t[:idx3], t[idx_low:]))
                    v = np.concatenate((v[:idx3], v[idx_low:]))
                    # Rolls time & voltage arrays so that point of chosen minimum t is at index 0
                    t = np.roll(t, -idx3)
                    v = np.roll(v, -idx3)
                    if len(t) >= idx_high:
                        # Removes points after chosen point of maximum t in time & voltage arrays
                        t = t[:idx_high]
                        v = v[:idx_high]
                        # Sums time & voltage arrays
                        tsum += t
                        vsum += v
                        n += 1
            except Exception:
                pass
    # Finds average time & voltage arrays
    t_avg = tsum / n
    v_avg = vsum / n
    v_avg = v_avg / max(v_avg)

    # Plots average waveform & saves image
    plt.plot(t_avg, v_avg)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Voltage')
    plt.title('Average Waveform (' + delay_name + ', ' + shaping_name + ')')
    plt.savefig(save_file / str('avg_waveform_double_' + str(int(fsps_new / 1e6)) + ' Msps_' + delay_folder + "_" +
                                shaping + '.png'), dpi=360)

    # Saves average waveform data
    file_name = dest_path / 'hist_data_double' / str('avg_waveform_' + str(int(fsps_new / 1e6)) + ' Msps_' +
                                                     delay_folder + "_" + shaping + '.txt')
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


# Creates p3 double folder names
def initialize_folders(date, filter_band):
    gen_path = Path(r'/Volumes/TOSHIBA EXT/data/watchman')
    save_path = Path(str(gen_path / '%08d_watchman_spe/waveforms/%s') % (date, filter_band))
    data_path = Path(save_path / 'd2')
    dest_path = Path(save_path / 'd3')
    filt_path1 = Path(dest_path / 'rt_1_double')
    filt_path2 = Path(dest_path / 'rt_2_double')
    filt_path4 = Path(dest_path / 'rt_4_double')
    filt_path8 = Path(dest_path / 'rt_8_double')

    return gen_path, save_path, data_path, dest_path, filt_path1, filt_path2, filt_path4, filt_path8


# Creates p3 double folders
def make_folders(dest_path, filt_path1, filt_path2, filt_path4, filt_path8, fsps_new, delay_folder):
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
    if not os.path.exists(Path(filt_path1 / 'raw' / delay_folder)):
        print('Creating rt 1 raw delay folder')
        os.mkdir(Path(filt_path1 / 'raw' / delay_folder))
    if not os.path.exists(Path(filt_path2 / 'raw' / delay_folder)):
        print('Creating rt 2 raw delay folder')
        os.mkdir(Path(filt_path2 / 'raw' / delay_folder))
    if not os.path.exists(Path(filt_path4 / 'raw' / delay_folder)):
        print('Creating rt 4 raw delay folder')
        os.mkdir(Path(filt_path4 / 'raw' / delay_folder))
    if not os.path.exists(Path(filt_path8 / 'raw' / delay_folder)):
        print('Creating rt 8 raw delay folder')
        os.mkdir(Path(filt_path8 / 'raw' / delay_folder))
    if not os.path.exists(Path(filt_path1 / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps')) / delay_folder):
        print('Creating rt 1 downsampled delay folder')
        os.mkdir(Path(filt_path1 / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps')) / delay_folder)
    if not os.path.exists(Path(filt_path2 / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps')) / delay_folder):
        print('Creating rt 2 downsampled delay folder')
        os.mkdir(Path(filt_path2 / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps')) / delay_folder)
    if not os.path.exists(Path(filt_path4 / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps')) / delay_folder):
        print('Creating rt 4 downsampled delay folder')
        os.mkdir(Path(filt_path4 / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') / delay_folder))
    if not os.path.exists(Path(filt_path8 / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps')) / delay_folder):
        print('Creating rt 8 downsampled delay folder')
        os.mkdir(Path(filt_path8 / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') / delay_folder))
    if not os.path.exists(Path(filt_path1 / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps')) / delay_folder):
        print('Creating rt 1 digitized delay folder')
        os.mkdir(Path(filt_path1 / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') / delay_folder))
    if not os.path.exists(Path(filt_path2 / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps')) / delay_folder):
        print('Creating rt 2 digitized delay folder')
        os.mkdir(Path(filt_path2 / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') / delay_folder))
    if not os.path.exists(Path(filt_path4 / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps')) / delay_folder):
        print('Creating rt 4 digitized delay folder')
        os.mkdir(Path(filt_path4 / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') / delay_folder))
    if not os.path.exists(Path(filt_path8 / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps')) / delay_folder):
        print('Creating rt 8 digitized delay folder')
        os.mkdir(Path(filt_path8 / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') / delay_folder))
    if not os.path.exists(Path(Path(dest_path / 'rt_1_single_2') / 'raw')):
        print('Creating single rt 1 raw folder')
        os.mkdir(Path(Path(dest_path / 'rt_1_single_2') / 'raw'))
    if not os.path.exists(Path(Path(dest_path / 'rt_2_single_2') / 'raw')):
        print('Creating single rt 2 raw folder')
        os.mkdir(Path(Path(dest_path / 'rt_2_single_2') / 'raw'))
    if not os.path.exists(Path(Path(dest_path / 'rt_4_single_2') / 'raw')):
        print('Creating single rt 4 raw folder')
        os.mkdir(Path(Path(dest_path / 'rt_4_single_2') / 'raw'))
    if not os.path.exists(Path(Path(dest_path / 'rt_8_single_2') / 'raw')):
        print('Creating single rt 8 raw folder')
        os.mkdir(Path(Path(dest_path / 'rt_8_single_2') / 'raw'))
    if not os.path.exists(Path(Path(dest_path / 'rt_1_single_2') / str('downsampled_' + str(int(fsps_new / 1e6)) +
                                                                       '_Msps'))):
        print('Creating single rt 1 downsampled folder')
        os.mkdir(Path(Path(dest_path / 'rt_1_single_2') / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps')))
    if not os.path.exists(Path(Path(dest_path / 'rt_2_single_2') / str('downsampled_' + str(int(fsps_new / 1e6)) +
                                                                       '_Msps'))):
        print('Creating single rt 2 downsampled folder')
        os.mkdir(Path(Path(dest_path / 'rt_2_single_2') / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps')))
    if not os.path.exists(Path(Path(dest_path / 'rt_4_single_2') / str('downsampled_' + str(int(fsps_new / 1e6)) +
                                                                       '_Msps'))):
        print('Creating single rt 4 downsampled folder')
        os.mkdir(Path(Path(dest_path / 'rt_4_single_2') / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps')))
    if not os.path.exists(Path(Path(dest_path / 'rt_8_single_2') / str('downsampled_' + str(int(fsps_new / 1e6)) +
                                                                       '_Msps'))):
        print('Creating single rt 8 downsampled folder')
        os.mkdir(Path(Path(dest_path / 'rt_8_single_2') / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps')))
    if not os.path.exists(Path(Path(dest_path / 'rt_1_single_2') / str('digitized_' + str(int(fsps_new / 1e6)) +
                                                                       '_Msps'))):
        print('Creating single rt 1 digitized folder')
        os.mkdir(Path(Path(dest_path / 'rt_1_single_2') / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps')))
    if not os.path.exists(Path(Path(dest_path / 'rt_2_single_2') / str('digitized_' + str(int(fsps_new / 1e6)) +
                                                                       '_Msps'))):
        print('Creating single rt 2 digitized folder')
        os.mkdir(Path(Path(dest_path / 'rt_2_single_2') / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps')))
    if not os.path.exists(Path(Path(dest_path / 'rt_4_single_2') / str('digitized_' + str(int(fsps_new / 1e6)) +
                                                                       '_Msps'))):
        print('Creating single rt 4 digitized folder')
        os.mkdir(Path(Path(dest_path / 'rt_4_single_2') / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps')))
    if not os.path.exists(Path(Path(dest_path / 'rt_8_single_2') / str('digitized_' + str(int(fsps_new / 1e6)) +
                                                                       '_Msps'))):
        print('Creating single rt 8 digitized folder')
        os.mkdir(Path(Path(dest_path / 'rt_8_single_2') / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps')))
    if not os.path.exists(Path(dest_path / 'hist_data_single')):
        print('Creating histogram data folder')
        os.mkdir(Path(dest_path / 'hist_data_single'))
    if not os.path.exists(Path(dest_path / 'plots')):
        print('Creating plots folder')
        os.mkdir(Path(dest_path / 'plots'))
    if not os.path.exists(Path(dest_path / 'unusable_data')):
        print('Creating unusable data folder')
        os.mkdir(Path(dest_path / 'unusable_data'))


# Makes arrays of existing single and double spe files
def initial_arrays(filt_path1_s, delay_path1):
    single_file_array = np.array([])
    double_file_array = np.array([])

    print('Checking existing single spe files...')
    for i in range(99999):                                  # Makes array of all spe file names
        file_name = 'D3--waveforms--%05d.txt' % i
        if os.path.isfile(filt_path1_s / file_name):
            single_file_array = np.append(single_file_array, i)

    print('Checking existing double spe files...')
    for filename in os.listdir(delay_path1):                # Checks for existing double spe files
        print(filename, 'is a file')
        files_added = filename[15:27]
        double_file_array = np.append(double_file_array, files_added)

    return single_file_array, double_file_array


# Copies single spe waveforms with 1x, 2x, 4x, and 8x initial rise times to d3 folder
def copy_s_waveforms(single_file_array, data_path, dest_path, nhdr):
    for item in single_file_array:
        file_name1 = str(data_path / 'rt_1_single' / 'D2--waveforms--%s.txt') % item
        file_name2 = str(data_path / 'rt_2_single' / 'D2--waveforms--%s.txt') % item
        file_name4 = str(data_path / 'rt_4_single' / 'D2--waveforms--%s.txt') % item
        file_name8 = str(data_path / 'rt_8_single' / 'D2--waveforms--%s.txt') % item
        save_name1 = str(dest_path / 'rt_1_single_2' / 'raw' / 'D3--waveforms--%s.txt') % item
        save_name2 = str(dest_path / 'rt_2_single_2' / 'raw' / 'D3--waveforms--%s.txt') % item
        save_name4 = str(dest_path / 'rt_4_single_2' / 'raw' / 'D3--waveforms--%s.txt') % item
        save_name8 = str(dest_path / 'rt_8_single_2' / 'raw' / 'D3--waveforms--%s.txt') % item

        if os.path.isfile(file_name1):
            if os.path.isfile(save_name1):
                print('File #%s in rt_1 folder' % item)
            else:
                t, v, hdr = rw(file_name1, nhdr)
                ww(t, v, save_name1, hdr)
                print('File #%s in rt_1 folder' % item)

        if os.path.isfile(file_name2):
            if os.path.isfile(save_name2):
                print('File #%s in rt_2 folder' % item)
            else:
                t, v, hdr = rw(file_name2, nhdr)
                ww(t, v, save_name2, hdr)
                print('File #%s in rt_2 folder' % item)

        if os.path.isfile(file_name4):
            if os.path.isfile(save_name4):
                print('File #%s in rt_4 folder' % item)
            else:
                t, v, hdr = rw(file_name4, nhdr)
                ww(t, v, save_name4, hdr)
                print('File #%s in rt_4 folder' % item)

        if os.path.isfile(file_name8):
            if os.path.isfile(save_name8):
                print('File #%s in rt_8 folder' % item)
            else:
                t, v, hdr = rw(file_name8, nhdr)
                ww(t, v, save_name8, hdr)
                print('File #%s in rt_8 folder' % item)


# Copies double spe waveforms with 1x, 2x, 4x, and 8x initial rise times to d3 folder
def copy_d_waveforms(double_file_array, data_path, filt_path1, filt_path2, filt_path4, filt_path8, delay_folder, nhdr):
    for item in double_file_array:
        file_name1 = str(data_path / 'rt_1_double' / delay_folder / 'D2--waveforms--%s.txt') % item
        file_name2 = str(data_path / 'rt_2_double' / delay_folder / 'D2--waveforms--%s.txt') % item
        file_name4 = str(data_path / 'rt_4_double' / delay_folder / 'D2--waveforms--%s.txt') % item
        file_name8 = str(data_path / 'rt_8_double' / delay_folder / 'D2--waveforms--%s.txt') % item
        save_name1 = str(filt_path1 / 'raw' / delay_folder / 'D3--waveforms--%s.txt') % item
        save_name2 = str(filt_path2 / 'raw' / delay_folder / 'D3--waveforms--%s.txt') % item
        save_name4 = str(filt_path4 / 'raw' / delay_folder / 'D3--waveforms--%s.txt') % item
        save_name8 = str(filt_path8 / 'raw' / delay_folder / 'D3--waveforms--%s.txt') % item

        if os.path.isfile(file_name1):
            if os.path.isfile(save_name1):
                print('File #%s in double_spe folder' % item)
            else:
                t, v, hdr = rw(file_name1, nhdr)
                ww(t, v, save_name1, hdr)
                print('File #%s in double_spe folder' % item)

        if os.path.isfile(file_name2):
            if os.path.isfile(save_name2):
                print('File #%s in double_spe_2 folder' % item)
            else:
                t, v, hdr = rw(file_name2, nhdr)
                ww(t, v, save_name2, hdr)
                print('File #%s in double_spe_2 folder' % item)

        if os.path.isfile(file_name4):
            if os.path.isfile(save_name4):
                print('File #%s in double_spe_4 folder' % item)
            else:
                t, v, hdr = rw(file_name4, nhdr)
                ww(t, v, save_name4, hdr)
                print('File #%s in double_spe_4 folder' % item)

        if os.path.isfile(file_name8):
            if os.path.isfile(save_name8):
                print('File #%s in double_spe_8 folder' % item)
            else:
                t, v, hdr = rw(file_name8, nhdr)
                ww(t, v, save_name8, hdr)
                print('File #%s in double_spe_8 folder' % item)
