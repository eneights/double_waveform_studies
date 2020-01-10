import os
import csv
import datetime
import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import norm
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


# Creates text file with time of beginning of spe, time of end of spe, charge, amplitude, and fwhm for a single spe file
def save_calculations_s(dest_path, item, t1, t2, charge, amplitude, fwhm, shaping, fsps_new):
    file_name = str(dest_path / 'calculations_single' / str(str(int(fsps_new / 1e6)) + '_Msps') / shaping /
                    'D3--waveforms--%05d.txt') % item
    myfile = open(file_name, 'w')
    myfile.write('t1,' + str(t1))
    myfile.write('\nt2,' + str(t2))
    myfile.write('\ncharge,' + str(charge))
    myfile.write('\namplitude,' + str(amplitude))
    myfile.write('\nfwhm,' + str(fwhm))
    myfile.close()


# Creates text file with time of beginning of spe, time of end of spe, charge, amplitude, fwhm for a double spe file
def save_calculations_d(dest_path, delay_folder, item, t1, t2, charge, amplitude, fwhm, shaping, fsps_new):
    file_name = str(dest_path / 'calculations_double' / str(str(int(fsps_new / 1e6)) + '_Msps') / delay_folder / shaping
                    / 'D3--waveforms--%s.txt') % item
    myfile = open(file_name, 'w')
    myfile.write('t1,' + str(t1))
    myfile.write('\nt2,' + str(t2))
    myfile.write('\ncharge,' + str(charge))
    myfile.write('\namplitude,' + str(amplitude))
    myfile.write('\nfwhm,' + str(fwhm))
    myfile.close()


# Reads calculation file
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

    return t1, t2, charge, amp, fwhm


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


# Creates text file with data from an array
def write_hist_data_s(array, dest_path, name):
    array = np.sort(array)
    file_name = Path(Path(dest_path) / 'hist_data_single' / name)

    myfile = open(file_name, 'w')
    for item in array:  # Writes an array item on each line of file
        myfile.write(str(item) + '\n')
    myfile.close()


# Creates text file with data from an array
def write_hist_data_d(array, dest_path, name):
    array = np.sort(array)
    file_name = Path(Path(dest_path) / 'hist_data_double' / name)

    myfile = open(file_name, 'w')
    for item in array:  # Writes an array item on each line of file
        myfile.write(str(item) + '\n')
    myfile.close()


def read_hist_file(filename, fsps_new):
    array = np.array([])

    myfile = open(filename, 'r')                # Opens file
    for line in myfile:                         # Reads values & saves in an array
        line = line.strip()
        line = float(line)
        array = np.append(array, line)          # Closes histogram file
    myfile.close()

    return array


# HISTOGRAMS


# Defines Gaussian function (a is amplitude, b is mean, c is standard deviation)
def func(x, a, b, c):
    gauss = a * np.exp(-(x - b) ** 2.0 / (2 * c ** 2))
    return gauss


# Finds Gaussian fit of array
def gauss_fit(array, bins, n):
    b_est, c_est = norm.fit(array)      # Calculates mean & standard deviation based on entire array
    range_min1 = b_est - c_est          # Calculates lower limit of Gaussian fit (1sigma estimation)
    range_max1 = b_est + c_est          # Calculates upper limit of Gaussian fit (1sigma estimation)

    bins_range1 = np.linspace(range_min1, range_max1, 10000)    # Creates array of bins between upper & lower limits
    n_range1 = np.interp(bins_range1, bins, n)              # Interpolates & creates array of y axis values
    guess1 = [1, float(b_est), float(c_est)]                # Defines guess for values of a, b & c in Gaussian fit
    popt1, pcov1 = curve_fit(func, bins_range1, n_range1, p0=guess1, maxfev=100000)      # Finds Gaussian fit
    mu1 = float(format(popt1[1], '.2e'))                        # Calculates mean based on 1sigma guess
    sigma1 = np.abs(float(format(popt1[2], '.2e')))     # Calculates standard deviation based on 1sigma estimation
    range_min2 = mu1 - 2 * sigma1                       # Calculates lower limit of Gaussian fit (2sigma)
    range_max2 = mu1 + 2 * sigma1                       # Calculates upper limit of Gaussian fit (2sigma)
    bins_range2 = np.linspace(range_min2, range_max2, 10000)    # Creates array of bins between upper & lower limits
    n_range2 = np.interp(bins_range2, bins, n)          # Interpolates & creates array of y axis values
    guess2 = [1, mu1, sigma1]                           # Defines guess for values of a, b & c in Gaussian fit
    popt2, pcov2 = curve_fit(func, bins_range2, n_range2, p0=guess2, maxfev=100000)      # Finds Gaussian fit

    return bins_range2, popt2, pcov2


# Creates histogram given an array
def plot_histogram(array, dest_path, nbins, xaxis, title, units, filename, type):

    path = Path(Path(dest_path) / 'plots')
    n, bins, patches = plt.hist(array, nbins)           # Plots histogram
    bins = np.delete(bins, len(bins) - 1)
    bins_diff = bins[1] - bins[0]
    bins = np.linspace(bins[0] + bins_diff / 2, bins[len(bins) - 1] + bins_diff / 2, len(bins))

    bins_range, popt, pcov = gauss_fit(array, bins, n)                  # Finds Gaussian fit
    plt.plot(bins_range, func(bins_range, *popt), color='red')          # Plots Gaussian fit (mean +/- 2sigma)

    mu2 = float(format(popt[1], '.2e'))                 # Calculates mean
    sigma2 = np.abs(float(format(popt[2], '.2e')))      # Calculates standard deviation

    plt.xlabel(xaxis + ' (' + units + ')')
    plt.title(title + ' of SPE\n mean: ' + str(mu2) + ' ' + units + ', SD: ' + str(sigma2) + ' ' + units)
    plt.savefig(path / str(filename + '.png'), dpi=360)         # Plots histogram with Gaussian fit

    if type == 'single':
        write_hist_data_s(array, dest_path, filename + '.txt')
    else:
        write_hist_data_d(array, dest_path, filename + '.txt')

    plt.close()

    mean = mu2

    return mean


# Creates two histograms on top of each other given two arrays
def plot_histograms(array1, array2, dest_path, nbins, xaxis, title, units, filename):

    path = Path(Path(dest_path) / 'plots')

    n1, bins1, patches1 = plt.hist(array1, nbins)               # Plots histogram 1
    bins1 = np.delete(bins1, len(bins1) - 1)
    bins_diff1 = bins1[1] - bins1[0]
    bins1 = np.linspace(bins1[0] + bins_diff1 / 2, bins1[len(bins1) - 1] + bins_diff1 / 2, len(bins1))

    bins_range1, popt1, pcov1 = gauss_fit(array1, bins1, n1)            # Finds Gaussian fit of histogram 1
    plt.plot(bins_range1, func(bins_range1, *popt1), color='red')       # Plots Gaussian fit (mean +/- 2sigma) of hist 1

    mu2_1 = float(format(popt1[1], '.2e'))                  # Calculates mean of hist 1
    sigma2_1 = np.abs(float(format(popt1[2], '.2e')))       # Calculates standard deviation of hist 1

    n2, bins2, patches2 = plt.hist(array2, nbins)               # Plots histogram 2
    bins2 = np.delete(bins2, len(bins2) - 1)
    bins_diff2 = bins2[1] - bins2[0]
    bins2 = np.linspace(bins2[0] + bins_diff2 / 2, bins2[len(bins2) - 1] + bins_diff2 / 2, len(bins2))

    bins_range2, popt2, pcov2 = gauss_fit(array2, bins2, n2)            # Finds Gaussian fit of histogram 2
    plt.plot(bins_range2, func(bins_range2, *popt2), color='green')     # Plots Gaussian fit (mean +/- 2sigma) of hist 2

    mu2_2 = float(format(popt2[1], '.2e'))                  # Calculates mean of hist 2
    sigma2_2 = np.abs(float(format(popt2[2], '.2e')))       # Calculates standard deviation of hist 2

    plt.xlabel(xaxis + ' (' + units + ')')
    plt.title(title + ' of SPE\n mean (single): ' + str(mu2_1) + ' ' + units + ', SD (single): ' + str(sigma2_1) +
              ' ' + units + '\n mean (double): ' + str(mu2_2) + ' ' + units + ', SD (double): ' + str(sigma2_2) +
              ' ' + units, fontsize='medium')
    plt.savefig(path / str(filename + '.png'), dpi=360)         # Plots histogram with Gaussian fit

    plt.close()


# Plots histograms for each type of calculation array
def make_hist(charge_array, amplitude_array, fwhm_array, dest_path, bins, version, type):
    mean_charge = plot_histogram(charge_array, dest_path, bins, 'Charge', 'Charge', 's*bit/ohm', 'charge_' + version,
                                 type)
    mean_amp = plot_histogram(amplitude_array, dest_path, bins, 'Voltage', 'Amplitude', 'bits', 'amplitude_' + version,
                              type)
    mean_fwhm = plot_histogram(fwhm_array, dest_path, bins, 'Time', 'FWHM', 's', 'fwhm_' + version, type)

    return mean_charge, mean_amp, mean_fwhm


# Plots single and double histograms for each type of calculation array on same plot
def make_double_hist(charge_array_s, amplitude_array_s, fwhm_array_s, charge_array_d, amplitude_array_d, fwhm_array_d,
                     dest_path, bins, version):
    plot_histograms(charge_array_s, charge_array_d, dest_path, bins, 'Charge', 'Charge', 's*bit/ohm', 'charge_' +
                    version)
    plot_histograms(amplitude_array_s, amplitude_array_d, dest_path, bins, 'Voltage', 'Amplitude', 'bits',
                    'amplitude_' + version)
    plot_histograms(fwhm_array_s, fwhm_array_d, dest_path, bins, 'Time', 'FWHM', 's',
                    'fwhm_' + version)


# Plots histograms for each calculation array
def p3_hist(dest_path, delay_folder, charge_array_s_1, charge_array_s_2, charge_array_s_4, charge_array_s_8,
            amplitude_array_s_1, amplitude_array_s_2, amplitude_array_s_4, amplitude_array_s_8, fwhm_array_s_1,
            fwhm_array_s_2, fwhm_array_s_4, fwhm_array_s_8, charge_array_d_1, charge_array_d_2, charge_array_d_4,
            charge_array_d_8, amplitude_array_d_1, amplitude_array_d_2, amplitude_array_d_4, amplitude_array_d_8,
            fwhm_array_d_1, fwhm_array_d_2, fwhm_array_d_4, fwhm_array_d_8, fsps_new):
    print('Creating histograms')
    mean_charge_s_1, mean_amp_s_1, mean_fwhm_s_1 = \
        make_hist(charge_array_s_1, amplitude_array_s_1, fwhm_array_s_1, dest_path, 75, str(int(fsps_new / 1e6)) +
                  '_Msps_rt_1_single', 'single')
    mean_charge_s_2, mean_amp_s_2, mean_fwhm_s_2 = \
        make_hist(charge_array_s_2, amplitude_array_s_2, fwhm_array_s_2, dest_path, 75, str(int(fsps_new / 1e6)) +
                  '_Msps_rt_2_single', 'single')
    mean_charge_s_4, mean_amp_s_4, mean_fwhm_s_4 = \
        make_hist(charge_array_s_4, amplitude_array_s_4, fwhm_array_s_4, dest_path, 75, str(int(fsps_new / 1e6)) +
                  '_Msps_rt_4_single', 'single')
    mean_charge_s_8, mean_amp_s_8, mean_fwhm_s_8 = \
        make_hist(charge_array_s_8, amplitude_array_s_8, fwhm_array_s_8, dest_path, 75, str(int(fsps_new / 1e6)) +
                  '_Msps_rt_8_single', 'single')
    mean_charge_d_1, mean_amp_d_1, mean_fwhm_d_1 = \
        make_hist(charge_array_d_1, amplitude_array_d_1, fwhm_array_d_1, dest_path, 75, str(int(fsps_new / 1e6)) +
                  '_Msps_rt_1_double_' + delay_folder, 'double')
    mean_charge_d_2, mean_amp_d_2, mean_fwhm_d_2 = \
        make_hist(charge_array_d_2, amplitude_array_d_2, fwhm_array_d_2, dest_path, 75, str(int(fsps_new / 1e6)) +
                  '_Msps_rt_2_double_' + delay_folder, 'double')
    mean_charge_d_4, mean_amp_d_4, mean_fwhm_d_4 = \
        make_hist(charge_array_d_4, amplitude_array_d_4, fwhm_array_d_4, dest_path, 75, str(int(fsps_new / 1e6)) +
                  '_Msps_rt_4_double_' + delay_folder, 'double')
    mean_charge_d_8, mean_amp_d_8, mean_fwhm_d_8 = \
        make_hist(charge_array_d_8, amplitude_array_d_8, fwhm_array_d_8, dest_path, 75, str(int(fsps_new / 1e6)) +
                  '_Msps_rt_8_double_' + delay_folder, 'double')
    make_double_hist(charge_array_s_1, amplitude_array_s_1, fwhm_array_s_1, charge_array_d_1, amplitude_array_d_1,
                     fwhm_array_d_1, dest_path, 75, str(int(fsps_new / 1e6)) + '_Msps_rt_1_' + delay_folder)
    make_double_hist(charge_array_s_2, amplitude_array_s_2, fwhm_array_s_2, charge_array_d_2, amplitude_array_d_2,
                     fwhm_array_d_2, dest_path, 75, str(int(fsps_new / 1e6)) + '_Msps_rt_2_' + delay_folder)
    make_double_hist(charge_array_s_4, amplitude_array_s_4, fwhm_array_s_4, charge_array_d_4, amplitude_array_d_4,
                     fwhm_array_d_4, dest_path, 75, str(int(fsps_new / 1e6)) + '_Msps_rt_4_' + delay_folder)
    make_double_hist(charge_array_s_8, amplitude_array_s_8, fwhm_array_s_8, charge_array_d_8, amplitude_array_d_8,
                     fwhm_array_d_8, dest_path, 75, str(int(fsps_new / 1e6)) + '_Msps_rt_8_' + delay_folder)

    return mean_charge_s_1, mean_amp_s_1, mean_fwhm_s_1, mean_charge_s_2, mean_amp_s_2, mean_fwhm_s_2, mean_charge_s_4, \
           mean_amp_s_4, mean_fwhm_s_4, mean_charge_s_8, mean_amp_s_8, mean_fwhm_s_8, mean_charge_d_1, mean_amp_d_1, \
           mean_fwhm_d_1, mean_charge_d_2, mean_amp_d_2, mean_fwhm_d_2, mean_charge_d_4, mean_amp_d_4, mean_fwhm_d_4,\
           mean_charge_d_8, mean_amp_d_8, mean_fwhm_d_8


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


# CALCULATIONS


# Returns time when spe waveform begins and time when spe waveform ends
def calculate_t1_t2(t, v):
    idx1 = np.inf
    idx2 = np.inf
    idx3 = np.inf

    for i in range(len(v)):
        if v[i] <= 0.1 * min(v):
            idx1 = i
            break
        else:
            continue

    if idx1 == np.inf:
        return 0, -1
    else:
        for i in range(idx1, len(v)):
            if v[i] == min(v):
                idx2 = i
                break
            else:
                continue
        if idx2 == np.inf:
            return 0, -1
        else:
            for i in range(len(v) - 1, idx2, -1):
                if v[i] <= 0.1 * min(v):
                    idx3 = i
                    break
                else:
                    continue

            t1 = t[idx1]                    # Finds time of beginning of spe
            t2 = t[idx3]                    # Finds time of end of spe

            return t1, t2


# Returns the average baseline (baseline noise level)
def calculate_average(t, v):
    v_sum = 0
    t1, t2 = calculate_t1_t2(t, v)

    try:
        for i in range(int(np.where(t == t1)[0] - (.1 * len(t)))):
            v_sum += v[i]

        average = v_sum / (int(np.where(t == t1)[0] - (.1 * len(t))))

        return average

    except:
        return np.inf


# Returns charge of spe (as a positive value)
def calculate_charge(t, v, r):
    vsum = 0
    tvals = np.linspace(t[0], t[len(t) - 1], 5000)      # Creates array of times over entire timespan
    vvals = np.interp(tvals, t, v)                      # Interpolates & creates array of voltages over entire timespan

    for i in range(len(tvals)):                         # Calculates sum of all voltages in full timespan
        vsum += vvals[i]
    charge = -1 * (tvals[len(tvals) - 1]) * vsum / (len(tvals) * r)     # Calculates charge

    return charge


# Returns the amplitude of spe as a positive value (minimum voltage)
def calculate_amp(t, v):
    avg = calculate_average(t, v)       # Calculates value of baseline voltage
    amp = avg - np.amin(v)              # Calculates max amplitude

    return amp


# Returns the full width half max (FWHM) of spe
def calculate_fwhm(t, v):
    time1 = np.inf
    time2 = np.inf

    t1, t2 = calculate_t1_t2(t, v)                      # Calculates start and end times of spe
    avg = calculate_average(t, v)                       # Calculates average baseline
    half_max = ((min(v) - avg) / 2).item()              # Calculates 50% max value

    tvals1 = np.linspace(t1, t[np.where(v == min(v))[0][0]], 2500)
    vvals1 = np.interp(tvals1, t, v)
    tvals2 = np.linspace(t[np.where(v == min(v))[0][0]], t2, 2500)
    vvals2 = np.interp(tvals2, t, v)

    for i in range(len(vvals1)):
        if vvals1[i] <= half_max:
            time1 = tvals1[i]
            break
        else:
            continue

    for i in range(len(vvals2) - 1, 0, -1):
        if vvals2[i] <= half_max:
            time2 = tvals2[i]
            break
        else:
            continue

    half_max_time = time2 - time1

    return half_max_time


# P3_DOUBLE_STUDIES


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
    if not os.path.exists(Path(Path(dest_path / 'rt_1_single_2'))):
        print('Creating single rt 1 folder')
        os.mkdir(Path(Path(dest_path / 'rt_1_single_2')))
    if not os.path.exists(Path(Path(dest_path / 'rt_2_single_2'))):
        print('Creating single rt 2 folder')
        os.mkdir(Path(Path(dest_path / 'rt_2_single_2')))
    if not os.path.exists(Path(Path(dest_path / 'rt_4_single_2'))):
        print('Creating single rt 4 folder')
        os.mkdir(Path(Path(dest_path / 'rt_4_single_2')))
    if not os.path.exists(Path(Path(dest_path / 'rt_8_single_2'))):
        print('Creating single rt 8 folder')
        os.mkdir(Path(Path(dest_path / 'rt_8_single_2')))
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
        print('Creating single histogram data folder')
        os.mkdir(Path(dest_path / 'hist_data_single'))
    if not os.path.exists(Path(dest_path / 'calculations_single')):
        print('Creating single calculations data folder')
        os.mkdir(Path(dest_path / 'calculations_single'))
    if not os.path.exists(Path(dest_path / 'calculations_single' / str(str(int(fsps_new / 1e6)) + '_Msps'))):
        print('Creating single digitized calculations data folder')
        os.mkdir(Path(dest_path / 'calculations_single' / str(str(int(fsps_new / 1e6)) + '_Msps')))
    if not os.path.exists(Path(dest_path / 'calculations_single' / str(str(int(fsps_new / 1e6)) + '_Msps') / 'rt_1')):
        print('Creating single rt_1 calculations folder')
        os.mkdir(Path(dest_path / 'calculations_single' / str(str(int(fsps_new / 1e6)) + '_Msps') / 'rt_1'))
    if not os.path.exists(Path(dest_path / 'calculations_single' / str(str(int(fsps_new / 1e6)) + '_Msps') / 'rt_2')):
        print('Creating single rt_2 calculations folder')
        os.mkdir(Path(dest_path / 'calculations_single' / str(str(int(fsps_new / 1e6)) + '_Msps') / 'rt_2'))
    if not os.path.exists(Path(dest_path / 'calculations_single' / str(str(int(fsps_new / 1e6)) + '_Msps') / 'rt_4')):
        print('Creating single rt_4 calculations folder')
        os.mkdir(Path(dest_path / 'calculations_single' / str(str(int(fsps_new / 1e6)) + '_Msps') / 'rt_4'))
    if not os.path.exists(Path(dest_path / 'calculations_single' / str(str(int(fsps_new / 1e6)) + '_Msps') / 'rt_8')):
        print('Creating single rt_8 calculations folder')
        os.mkdir(Path(dest_path / 'calculations_single' / str(str(int(fsps_new / 1e6)) + '_Msps') / 'rt_8'))
    if not os.path.exists(Path(dest_path / 'hist_data_double')):
        print('Creating single histogram data folder')
        os.mkdir(Path(dest_path / 'hist_data_double'))
    if not os.path.exists(Path(dest_path / 'calculations_double')):
        print('Creating double calculations data folder')
        os.mkdir(Path(dest_path / 'calculations_double'))
    if not os.path.exists(Path(dest_path / 'calculations_double' / str(str(int(fsps_new / 1e6)) + '_Msps'))):
        print('Creating double digitized calculations data folder')
        os.mkdir(Path(dest_path / 'calculations_double' / str(str(int(fsps_new / 1e6)) + '_Msps')))
    if not os.path.exists(Path(dest_path / 'calculations_double' / str(str(int(fsps_new / 1e6)) + '_Msps') /
                               delay_folder)):
        print('Creating double calculations delay folder')
        os.mkdir(Path(dest_path / 'calculations_double' / str(str(int(fsps_new / 1e6)) + '_Msps') / delay_folder))
    if not os.path.exists(Path(dest_path / 'calculations_double' / str(str(int(fsps_new / 1e6)) + '_Msps') /
                               delay_folder / 'rt_1')):
        print('Creating double rt_1 calculations folder')
        os.mkdir(Path(dest_path / 'calculations_double' / str(str(int(fsps_new / 1e6)) + '_Msps') / delay_folder /
                      'rt_1'))
    if not os.path.exists(Path(dest_path / 'calculations_double' / str(str(int(fsps_new / 1e6)) + '_Msps') /
                               delay_folder / 'rt_2')):
        print('Creating double rt_2 calculations folder')
        os.mkdir(Path(dest_path / 'calculations_double' / str(str(int(fsps_new / 1e6)) + '_Msps') / delay_folder /
                      'rt_2'))
    if not os.path.exists(Path(dest_path / 'calculations_double' / str(str(int(fsps_new / 1e6)) + '_Msps') /
                               delay_folder / 'rt_4')):
        print('Creating double rt_4 calculations folder')
        os.mkdir(Path(dest_path / 'calculations_double' / str(str(int(fsps_new / 1e6)) + '_Msps') / delay_folder /
                      'rt_4'))
    if not os.path.exists(Path(dest_path / 'calculations_double' / str(str(int(fsps_new / 1e6)) + '_Msps') /
                               delay_folder / 'rt_8')):
        print('Creating double rt_8 calculations folder')
        os.mkdir(Path(dest_path / 'calculations_double' / str(str(int(fsps_new / 1e6)) + '_Msps') / delay_folder /
                      'rt_8'))
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

    print('Checking existing d3 single spe files...')
    for i in range(99999):                                  # Makes array of all spe file names
        file_name = 'D3--waveforms--%05d.txt' % i
        if os.path.isfile(filt_path1_s / file_name):
            single_file_array = np.append(single_file_array, i)

    print('Checking existing d3 double spe files...')
    for filename in os.listdir(delay_path1):                # Checks for existing double spe files
        files_added = filename[15:27]
        double_file_array = np.append(double_file_array, files_added)

    return single_file_array, double_file_array


# Makes arrays of existing single and double spe files
def initial_arrays_2(filt_path1_s, delay_path1):
    single_file_array = np.array([])
    double_file_array = np.array([])

    print('Checking existing d2 single spe files...')
    for i in range(99999):                                  # Makes array of all spe file names
        file_name = 'D2--waveforms--%05d.txt' % i
        if os.path.isfile(filt_path1_s / file_name):
            single_file_array = np.append(single_file_array, i)

    print('Checking existing d2 double spe files...')
    for filename in os.listdir(delay_path1):                # Checks for existing double spe files
        files_added = filename[15:27]
        double_file_array = np.append(double_file_array, files_added)

    return single_file_array, double_file_array


# Copies single spe waveforms with 1x, 2x, 4x, and 8x initial rise times to d3 folder
def copy_s_waveforms(single_file_array, single_file_array_2, data_path, dest_path, nhdr):
    for item in single_file_array:
        file_name1 = str(data_path / 'rt_1_single_2' / 'D2--waveforms--%05d.txt') % item
        file_name2 = str(data_path / 'rt_2_single_2' / 'D2--waveforms--%05d.txt') % item
        file_name4 = str(data_path / 'rt_4_single_2' / 'D2--waveforms--%05d.txt') % item
        file_name8 = str(data_path / 'rt_8_single_2' / 'D2--waveforms--%05d.txt') % item
        save_name1 = str(dest_path / 'rt_1_single_2' / 'raw' / 'D3--waveforms--%05d.txt') % item
        save_name2 = str(dest_path / 'rt_2_single_2' / 'raw' / 'D3--waveforms--%05d.txt') % item
        save_name4 = str(dest_path / 'rt_4_single_2' / 'raw' / 'D3--waveforms--%05d.txt') % item
        save_name8 = str(dest_path / 'rt_8_single_2' / 'raw' / 'D3--waveforms--%05d.txt') % item

        if os.path.isfile(file_name1):
            if os.path.isfile(save_name1):
                print('File #%05d in rt_1 folder' % item)
            else:
                t, v, hdr = rw(file_name1, nhdr)
                ww(t, v, save_name1, hdr)
                print('File #%05d in rt_1 folder' % item)

        if os.path.isfile(file_name2):
            if os.path.isfile(save_name2):
                print('File #%05d in rt_2 folder' % item)
            else:
                t, v, hdr = rw(file_name2, nhdr)
                ww(t, v, save_name2, hdr)
                print('File #%05d in rt_2 folder' % item)

        if os.path.isfile(file_name4):
            if os.path.isfile(save_name4):
                print('File #%05d in rt_4 folder' % item)
            else:
                t, v, hdr = rw(file_name4, nhdr)
                ww(t, v, save_name4, hdr)
                print('File #%05d in rt_4 folder' % item)

        if os.path.isfile(file_name8):
            if os.path.isfile(save_name8):
                print('File #%05d in rt_8 folder' % item)
            else:
                t, v, hdr = rw(file_name8, nhdr)
                ww(t, v, save_name8, hdr)
                print('File #%05d in rt_8 folder' % item)

        single_file_array_2 = np.append(single_file_array_2, item)

    return single_file_array_2


# Copies double spe waveforms with 1x, 2x, 4x, and 8x initial rise times to d3 folder
def copy_d_waveforms(double_file_array, double_file_array_2, data_path, filt_path1, filt_path2, filt_path4, filt_path8,
                     delay_folder, nhdr):
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

        double_file_array_2 = np.append(double_file_array_2, item)

    return double_file_array_2


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
def down_dig(single_file_array, double_file_array, filt_path1, filt_path2, filt_path4, filt_path8, dest_path,
             delay_folder, fsps, fsps_new, noise, nhdr):
    for item in single_file_array:
        file_name1 = str(dest_path / 'rt_1_single' / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') /
                         'D3--waveforms--%05d.txt') % item
        file_name2 = str(dest_path / 'rt_2_single' / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') /
                         'D3--waveforms--%05d.txt') % item
        file_name4 = str(dest_path / 'rt_4_single' / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') /
                         'D3--waveforms--%05d.txt') % item
        file_name8 = str(dest_path / 'rt_8_single' / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') /
                         'D3--waveforms--%05d.txt') % item
        save_name1 = str(dest_path / 'rt_1_single_2' / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') /
                         'D3--waveforms--%05d.txt') % item
        save_name2 = str(dest_path / 'rt_2_single_2' / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') /
                         'D3--waveforms--%05d.txt') % item
        save_name4 = str(dest_path / 'rt_4_single_2' / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') /
                         'D3--waveforms--%05d.txt') % item
        save_name8 = str(dest_path / 'rt_8_single_2' / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') /
                         'D3--waveforms--%05d.txt') % item

        if os.path.isfile(file_name1):
            if os.path.isfile(save_name1):
                print('File #%05d downsampled' % item)
            else:
                t, v, hdr = rw(file_name1, nhdr)
                ww(t, v, save_name1, hdr)
                print('File #%05d downsampled' % item)

        if os.path.isfile(file_name2):
            if os.path.isfile(save_name2):
                print('File #%05d downsampled' % item)
            else:
                t, v, hdr = rw(file_name2, nhdr)
                ww(t, v, save_name2, hdr)
                print('File #%05d downsampled' % item)

        if os.path.isfile(file_name4):
            if os.path.isfile(save_name4):
                print('File #%05d downsampled' % item)
            else:
                t, v, hdr = rw(file_name4, nhdr)
                ww(t, v, save_name4, hdr)
                print('File #%05d downsampled' % item)

        if os.path.isfile(file_name8):
            if os.path.isfile(save_name8):
                print('File #%05d downsampled' % item)
            else:
                t, v, hdr = rw(file_name8, nhdr)
                ww(t, v, save_name8, hdr)
                print('File #%05d downsampled' % item)

        file_name1 = str(dest_path / 'rt_1_single' / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') /
                         'D3--waveforms--%05d.txt') % item
        file_name2 = str(dest_path / 'rt_2_single' / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') /
                         'D3--waveforms--%05d.txt') % item
        file_name4 = str(dest_path / 'rt_4_single' / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') /
                         'D3--waveforms--%05d.txt') % item
        file_name8 = str(dest_path / 'rt_8_single' / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') /
                         'D3--waveforms--%05d.txt') % item
        save_name1 = str(dest_path / 'rt_1_single_2' / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') /
                         'D3--waveforms--%05d.txt') % item
        save_name2 = str(dest_path / 'rt_2_single_2' / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') /
                         'D3--waveforms--%05d.txt') % item
        save_name4 = str(dest_path / 'rt_4_single_2' / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') /
                         'D3--waveforms--%05d.txt') % item
        save_name8 = str(dest_path / 'rt_8_single_2' / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') /
                         'D3--waveforms--%05d.txt') % item

        if os.path.isfile(file_name1):
            if os.path.isfile(save_name1):
                print('File #%05d digitized' % item)
            else:
                t, v, hdr = rw(file_name1, nhdr)
                ww(t, v, save_name1, hdr)
                print('File #%05d digitized' % item)

        if os.path.isfile(file_name2):
            if os.path.isfile(save_name2):
                print('File #%05d digitized' % item)
            else:
                t, v, hdr = rw(file_name2, nhdr)
                ww(t, v, save_name2, hdr)
                print('File #%05d digitized' % item)

        if os.path.isfile(file_name4):
            if os.path.isfile(save_name4):
                print('File #%05d digitized' % item)
            else:
                t, v, hdr = rw(file_name4, nhdr)
                ww(t, v, save_name4, hdr)
                print('File #%05d digitized' % item)

        if os.path.isfile(file_name8):
            if os.path.isfile(save_name8):
                print('File #%05d digitized' % item)
            else:
                t, v, hdr = rw(file_name8, nhdr)
                ww(t, v, save_name8, hdr)
                print('File #%05d digitized' % item)

    for item in double_file_array:
        file_name1 = str(filt_path1 / 'raw' / delay_folder / 'D3--waveforms--%s.txt') % item
        file_name2 = str(filt_path2 / 'raw' / delay_folder / 'D3--waveforms--%s.txt') % item
        file_name4 = str(filt_path4 / 'raw' / delay_folder / 'D3--waveforms--%s.txt') % item
        file_name8 = str(filt_path8 / 'raw' / delay_folder / 'D3--waveforms--%s.txt') % item
        down_name1 = str(filt_path1 / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') / delay_folder /
                         'D3--waveforms--%s.txt') % item
        down_name2 = str(filt_path2 / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') / delay_folder /
                         'D3--waveforms--%s.txt') % item
        down_name4 = str(filt_path4 / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') / delay_folder /
                         'D3--waveforms--%s.txt') % item
        down_name8 = str(filt_path8 / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') / delay_folder /
                         'D3--waveforms--%s.txt') % item
        dig_name1 = str(filt_path1 / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') / delay_folder /
                        'D3--waveforms--%s.txt') % item
        dig_name2 = str(filt_path2 / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') / delay_folder /
                        'D3--waveforms--%s.txt') % item
        dig_name4 = str(filt_path4 / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') / delay_folder /
                        'D3--waveforms--%s.txt') % item
        dig_name8 = str(filt_path8 / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') / delay_folder /
                        'D3--waveforms--%s.txt') % item

        if os.path.isfile(down_name1) and os.path.isfile(down_name2) and os.path.isfile(down_name4) and \
                os.path.isfile(down_name8):
            print('File #%s downsampled' % item)
        else:
            if os.path.isfile(file_name1) and os.path.isfile(file_name2) and os.path.isfile(file_name4) and \
                    os.path.isfile(file_name8):
                print('Downsampling file #%s' % item)
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
            print('File #%s digitized' % item)
        else:
            if os.path.isfile(file_name1) and os.path.isfile(file_name2) and os.path.isfile(file_name4) and \
                    os.path.isfile(file_name8):
                print('Digitizing file #%s' % item)
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


# Checks if calculated values are possible or not
def check_if_impossible(t1, t2, charge, amp, fwhm):
    if (t2 <= t1 or charge <= 0 or amp <= 0 or fwhm <= 0 or charge == np.inf or charge == np.nan or
            amp == np.inf or amp == np.nan or fwhm == np.inf or fwhm == np.nan):
        return 'impossible'
    else:
        return 'ok'


# Creates empty arrays for calculations
def initialize_arrays():

    t1_array = np.array([])
    t2_array = np.array([])
    charge_array = np.array([])
    amplitude_array = np.array([])
    fwhm_array = np.array([])

    return t1_array, t2_array, charge_array, amplitude_array, fwhm_array


# Creates array for a calculation
def append_arrays(t1, t2, charge, amplitude, fwhm, t1_array, t2_array, charge_array, amplitude_array, fwhm_array):

    t1_array = np.append(t1_array, t1)
    t2_array = np.append(t2_array, t2)
    charge_array = np.append(charge_array, charge)
    amplitude_array = np.append(amplitude_array, amplitude)
    fwhm_array = np.append(fwhm_array, fwhm)

    return t1_array, t2_array, charge_array, amplitude_array, fwhm_array


# Removes single spe waveform from all spe folders
def remove_spe_s(rt_1_path_raw, rt_1_path_dig, rt_2_path_dig, rt_4_path_dig, rt_8_path_dig, rt_1_path_dow,
                 rt_2_path_dow, rt_4_path_dow, rt_8_path_dow, dest_path, number, nhdr, shaping, fsps_new):
    t, v, hdr = rw(str(Path(rt_1_path_raw) / 'D3--waveforms--%05d.txt') % number, nhdr)
    ww(t, v, str(dest_path / 'unusable_data' / 'D3--waveforms--%05d.txt') % number, hdr)
    if os.path.isfile(str(Path(rt_1_path_dig) / 'D3--waveforms--%05d.txt') % number):
        os.remove(str(Path(rt_1_path_dig) / 'D3--waveforms--%05d.txt') % number)
    if os.path.isfile(str(Path(rt_2_path_dig) / 'D3--waveforms--%05d.txt') % number):
        os.remove(str(Path(rt_2_path_dig) / 'D3--waveforms--%05d.txt') % number)
    if os.path.isfile(str(Path(rt_4_path_dig) / 'D3--waveforms--%05d.txt') % number):
        os.remove(str(Path(rt_4_path_dig) / 'D3--waveforms--%05d.txt') % number)
    if os.path.isfile(str(Path(rt_8_path_dig) / 'D3--waveforms--%05d.txt') % number):
        os.remove(str(Path(rt_8_path_dig) / 'D3--waveforms--%05d.txt') % number)
    if os.path.isfile(str(Path(rt_1_path_dow) / 'D3--waveforms--%05d.txt') % number):
        os.remove(str(Path(rt_1_path_dow) / 'D3--waveforms--%05d.txt') % number)
    if os.path.isfile(str(Path(rt_2_path_dow) / 'D3--waveforms--%05d.txt') % number):
        os.remove(str(Path(rt_2_path_dow) / 'D3--waveforms--%05d.txt') % number)
    if os.path.isfile(str(Path(rt_4_path_dow) / 'D3--waveforms--%05d.txt') % number):
        os.remove(str(Path(rt_4_path_dow) / 'D3--waveforms--%05d.txt') % number)
    if os.path.isfile(str(Path(rt_8_path_dow) / 'D3--waveforms--%05d.txt') % number):
        os.remove(str(Path(rt_8_path_dow) / 'D3--waveforms--%05d.txt') % number)
    if os.path.isfile(str(Path(dest_path) / 'calculations_single' / str(str(int(fsps_new / 1e6)) + '_Msps') / shaping /
                          'D3--waveforms--%05d.txt') % number):
        os.remove(str(Path(dest_path) / 'calculations_single' / str(str(int(fsps_new / 1e6)) + '_Msps') / shaping /
                      'D3--waveforms--%05d.txt') % number)


# Removes double spe waveform from all spe folders
def remove_spe_d(rt_1_path_raw, rt_1_path_dig, rt_2_path_dig, rt_4_path_dig, rt_8_path_dig, rt_1_path_dow,
                 rt_2_path_dow, rt_4_path_dow, rt_8_path_dow, dest_path, number, nhdr, delay_folder, shaping, fsps_new):
    t, v, hdr = rw(str(Path(rt_1_path_raw) / 'D3--waveforms--%s.txt') % number, nhdr)
    ww(t, v, str(Path(dest_path) / 'unusable_data' / 'D3--waveforms--%s.txt') % number, hdr)
    if os.path.isfile(str(Path(rt_1_path_dig) / 'D3--waveforms--%s.txt') % number):
        os.remove(str(Path(rt_1_path_dig) / 'D3--waveforms--%s.txt') % number)
    if os.path.isfile(str(Path(rt_2_path_dig) / 'D3--waveforms--%s.txt') % number):
        os.remove(str(Path(rt_2_path_dig) / 'D3--waveforms--%s.txt') % number)
    if os.path.isfile(str(Path(rt_4_path_dig) / 'D3--waveforms--%s.txt') % number):
        os.remove(str(Path(rt_4_path_dig) / 'D3--waveforms--%s.txt') % number)
    if os.path.isfile(str(Path(rt_8_path_dig) / 'D3--waveforms--%s.txt') % number):
        os.remove(str(Path(rt_8_path_dig) / 'D3--waveforms--%s.txt') % number)
    if os.path.isfile(str(Path(rt_1_path_dow) / 'D3--waveforms--%s.txt') % number):
        os.remove(str(Path(rt_1_path_dow) / 'D3--waveforms--%s.txt') % number)
    if os.path.isfile(str(Path(rt_2_path_dow) / 'D3--waveforms--%s.txt') % number):
        os.remove(str(Path(rt_2_path_dow) / 'D3--waveforms--%s.txt') % number)
    if os.path.isfile(str(Path(rt_4_path_dow) / 'D3--waveforms--%s.txt') % number):
        os.remove(str(Path(rt_4_path_dow) / 'D3--waveforms--%s.txt') % number)
    if os.path.isfile(str(Path(rt_8_path_dow) / 'D3--waveforms--%s.txt') % number):
        os.remove(str(Path(rt_8_path_dow) / 'D3--waveforms--%s.txt') % number)
    if os.path.isfile(str(Path(dest_path) / 'calculations_double' / str(str(int(fsps_new / 1e6)) + '_Msps') /
                          delay_folder / shaping / 'D3--waveforms--%s.txt') % number):
        os.remove(str(Path(dest_path) / 'calculations_double' / str(str(int(fsps_new / 1e6)) + '_Msps') /
                      delay_folder / shaping / 'D3--waveforms--%s.txt') % number)


# Calculates beginning & end times of spe waveform, charge, amplitude, fwhm, 10-90 & 20-80 rise times, 10-90 & 20-80
# fall times, and 10%, 20%, 80% & 90% jitter
def calculations(t, v, r):
    charge = calculate_charge(t, v, r)
    t1, t2 = calculate_t1_t2(t, v)
    amp = calculate_amp(t, v)
    fwhm = calculate_fwhm(t, v)

    return t1, t2, charge, amp, fwhm


# Reads calculations from an existing file and checks if they are possible values
def read_calculations(filename):
    t1, t2, charge, amplitude, fwhm = read_calc(filename)
    possibility = check_if_impossible(t1, t2, charge, amplitude, fwhm)

    return t1, t2, charge, amplitude, fwhm, possibility


# Removes spe file if values are impossible, appends values to arrays if not, and creates calculations file if it does
# not already exist
def create_arrays_s(calc_file, rt_1_path, rt_2_path, rt_4_path, rt_8_path, dest_path, number, t1_array, t2_array,
                    charge_array, amplitude_array, fwhm_array, t1, t2, charge, amplitude, fwhm, possibility, nhdr,
                    fsps_new, shaping):
    rt_1_path_raw = str(rt_1_path / 'raw')
    rt_1_path_dow = str(rt_1_path / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps'))
    rt_1_path_dig = str(rt_1_path / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps'))
    rt_2_path_dow = str(rt_2_path / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps'))
    rt_2_path_dig = str(rt_2_path / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps'))
    rt_4_path_dow = str(rt_4_path / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps'))
    rt_4_path_dig = str(rt_4_path / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps'))
    rt_8_path_dow = str(rt_8_path / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps'))
    rt_8_path_dig = str(rt_8_path / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps'))

    # Any spe waveform that returns impossible values is put into the not_spe folder
    if possibility == 'impossible' and not shaping == 'rt_8':
        print('Removing file #%05d' % number)
        remove_spe_s(rt_1_path_raw, rt_1_path_dig, rt_2_path_dig, rt_4_path_dig, rt_8_path_dig, rt_1_path_dow,
                     rt_2_path_dow, rt_4_path_dow, rt_8_path_dow, dest_path, number, nhdr, shaping, fsps_new)

    # All other spe waveforms' calculations are placed into arrays
    else:
        t1_array, t2_array, charge_array, amplitude_array, fwhm_array = \
            append_arrays(t1, t2, charge, amplitude, fwhm, t1_array, t2_array, charge_array, amplitude_array,
                          fwhm_array)
        if not os.path.isfile(calc_file):
            save_calculations_s(dest_path, number, t1, t2, charge, amplitude, fwhm, shaping, fsps_new)

    return t1_array, t2_array, charge_array, amplitude_array, fwhm_array


# Removes spe file if values are impossible, appends values to arrays if not, and creates calculations file if it does
# not already exist
def create_arrays_d(calc_file, rt_1_path, rt_2_path, rt_4_path, rt_8_path, dest_path, number, t1_array, t2_array,
                    charge_array, amplitude_array, fwhm_array, t1, t2, charge, amplitude, fwhm, possibility, nhdr,
                    fsps_new, delay_folder, shaping):
    rt_1_path_raw = str(rt_1_path / 'raw' / delay_folder)
    rt_1_path_dow = str(rt_1_path / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') / delay_folder)
    rt_1_path_dig = str(rt_1_path / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') / delay_folder)
    rt_2_path_dow = str(rt_2_path / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') / delay_folder)
    rt_2_path_dig = str(rt_2_path / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') / delay_folder)
    rt_4_path_dow = str(rt_4_path / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') / delay_folder)
    rt_4_path_dig = str(rt_4_path / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') / delay_folder)
    rt_8_path_dow = str(rt_8_path / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') / delay_folder)
    rt_8_path_dig = str(rt_8_path / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') / delay_folder)

    # Any spe waveform that returns impossible values is put into the not_spe folder
    if possibility == 'impossible' and not shaping == 'rt_8':
        print('Removing file #%s' % number)
        remove_spe_d(rt_1_path_raw, rt_1_path_dig, rt_2_path_dig, rt_4_path_dig, rt_8_path_dig, rt_1_path_dow,
                     rt_2_path_dow, rt_4_path_dow, rt_8_path_dow, dest_path, number, nhdr, delay_folder, shaping,
                     fsps_new)
        
    elif possibility == 'impossible' and shaping == 'rt_8':
        if not os.path.isfile(calc_file):
            save_calculations_d(dest_path, delay_folder, number, t1, t2, charge, amplitude, fwhm, shaping, fsps_new)

    # All other spe waveforms' calculations are placed into arrays
    else:
        t1_array, t2_array, charge_array, amplitude_array, fwhm_array = \
            append_arrays(t1, t2, charge, amplitude, fwhm, t1_array, t2_array, charge_array, amplitude_array,
                          fwhm_array)
        if not os.path.isfile(calc_file):
            save_calculations_d(dest_path, delay_folder, number, t1, t2, charge, amplitude, fwhm, shaping, fsps_new)

    return t1_array, t2_array, charge_array, amplitude_array, fwhm_array


# Calculates beginning & end times of spe waveform, charge, amplitude, and fwhm for each single spe file
# Returns arrays of beginning & end times of spe waveform, charge, amplitude, and fwhm
def make_arrays_s(save_shift, dest_path, array, nhdr, r, fsps_new, shaping):
    t1_array, t2_array, charge_array, amplitude_array, fwhm_array = initialize_arrays()

    for item in array:
        file_name1 = str(save_shift / 'D3--waveforms--%05d.txt') % item
        file_name2 = str(dest_path / 'calculations_single' / str(str(int(fsps_new / 1e6)) + '_Msps') / shaping /
                         'D3--waveforms--%05d.txt') % item

        if os.path.isfile(file_name1):
            # If the calculations were done previously, they are read from a file
            if os.path.isfile(file_name2):
                print("Reading calculations from file #%05d" % item)
                t1, t2, charge, amplitude, fwhm, possibility = read_calculations(file_name2)
            # If the calculations were not done yet, they are calculated
            else:
                print("Calculating file #%05d" % item)
                t, v, hdr = rw(file_name1, nhdr)                            # Shifted waveform file is read
                t1, t2, charge, amplitude, fwhm = calculations(t, v, r)     # Calculations are done
                possibility = check_if_impossible(t1, t2, charge, amplitude, fwhm)

            rt_1_path = Path(dest_path / 'rt_1_single_2')
            rt_2_path = Path(dest_path / 'rt_2_single_2')
            rt_4_path = Path(dest_path / 'rt_4_single_2')
            rt_8_path = Path(dest_path / 'rt_8_single_2')

            t1_array, t2_array, charge_array, amplitude_array, fwhm_array = \
                create_arrays_s(file_name2, rt_1_path, rt_2_path, rt_4_path, rt_8_path, dest_path, item, t1_array,
                                t2_array, charge_array, amplitude_array, fwhm_array, t1, t2, charge, amplitude, fwhm,
                                possibility, nhdr, fsps_new, shaping)

    return t1_array, t2_array, charge_array, amplitude_array, fwhm_array


# Calculates beginning & end times of spe waveform, charge, amplitude, and fwhm for each double spe file
# Returns arrays of beginning & end times of spe waveform, charge, amplitude, and fwhm
def make_arrays_d(save_shift, dest_path, delay_folder, array, nhdr, r, fsps_new, shaping):
    t1_array, t2_array, charge_array, amplitude_array, fwhm_array = initialize_arrays()

    for item in array:
        file_name1 = str(save_shift / 'D3--waveforms--%s.txt') % item
        file_name2 = str(dest_path / 'calculations_double' / str(str(int(fsps_new / 1e6)) + '_Msps') / delay_folder /
                         shaping / 'D3--waveforms--%s.txt') % item

        if os.path.isfile(file_name1):
            # If the calculations were done previously, they are read from a file
            if os.path.isfile(file_name2):
                print("Reading calculations from file #%s" % item)
                t1, t2, charge, amplitude, fwhm, possibility = read_calculations(file_name2)
            # If the calculations were not done yet, they are calculated
            else:
                print("Calculating file #%s" % item)
                t, v, hdr = rw(file_name1, nhdr)        # Shifted waveform file is read
                t1, t2, charge, amplitude, fwhm = calculations(t, v, r)             # Calculations are done
                possibility = check_if_impossible(t1, t2, charge, amplitude, fwhm)

            rt_1_path = Path(dest_path / 'rt_1_double')
            rt_2_path = Path(dest_path / 'rt_2_double')
            rt_4_path = Path(dest_path / 'rt_4_double')
            rt_8_path = Path(dest_path / 'rt_8_double')

            t1_array, t2_array, charge_array, amplitude_array, fwhm_array = \
                create_arrays_d(file_name2, rt_1_path, rt_2_path, rt_4_path, rt_8_path, dest_path, item, t1_array,
                                t2_array, charge_array, amplitude_array, fwhm_array, t1, t2, charge, amplitude, fwhm,
                                possibility, nhdr, fsps_new, delay_folder, shaping)

    return t1_array, t2_array, charge_array, amplitude_array, fwhm_array


# P3_DOUBLE_STUDIES_2


# Creates p3 double folder names
def initialize_folders2(date, filter_band):
    gen_path = Path(r'/Volumes/TOSHIBA EXT/data/watchman')
    save_path = Path(str(gen_path / '%08d_watchman_spe/waveforms/%s') % (date, filter_band))
    dest_path = Path(save_path / 'd3')
    hist_single = Path(dest_path / 'hist_data_single')
    hist_double = Path(dest_path / 'hist_data_double')

    return gen_path, save_path, dest_path, hist_single, hist_double


# Creates p3 double histogram file names
def initialize_names(hist_single, hist_double, shaping, fsps_new):
    amp_sing = Path(hist_single / str('amplitude_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping + '_single.txt'))
    charge_sing = Path(hist_single / str('charge_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping + '_single.txt'))
    fwhm_sing = Path(hist_single / str('fwhm_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping + '_single.txt'))

    amp_doub_no_delay = Path(hist_double / str('amplitude_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                               '_double_no_delay.txt'))
    amp_doub_05rt = Path(hist_double / str('amplitude_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                           '_double_0.5x_rt.txt'))
    amp_doub_1rt = Path(hist_double / str('amplitude_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                          '_double_1x_rt.txt'))
    amp_doub_15rt = Path(hist_double / str('amplitude_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                           '_double_1.5x_rt.txt'))
    amp_doub_2rt = Path(hist_double / str('amplitude_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                          '_double_2x_rt.txt'))
    amp_doub_25rt = Path(hist_double / str('amplitude_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                           '_double_2.5x_rt.txt'))
    amp_doub_3rt = Path(hist_double / str('amplitude_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                          '_double_3x_rt.txt'))
    amp_doub_35rt = Path(hist_double / str('amplitude_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                           '_double_3.5x_rt.txt'))
    amp_doub_4rt = Path(hist_double / str('amplitude_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                          '_double_4x_rt.txt'))
    amp_doub_45rt = Path(hist_double / str('amplitude_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                           '_double_4.5x_rt.txt'))
    amp_doub_5rt = Path(hist_double / str('amplitude_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                          '_double_5x_rt.txt'))
    amp_doub_55rt = Path(hist_double / str('amplitude_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                           '_double_5.5x_rt.txt'))
    amp_doub_6rt = Path(hist_double / str('amplitude_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                          '_double_6x_rt.txt'))
    amp_doub_40ns = Path(hist_double / str('amplitude_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                           '_double_40_ns.txt'))
    amp_doub_80ns = Path(hist_double / str('amplitude_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                           '_double_80_ns.txt'))

    charge_doub_no_delay = Path(hist_double / str('charge_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                                  '_double_no_delay.txt'))
    charge_doub_05rt = Path(hist_double / str('charge_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                              '_double_0.5x_rt.txt'))
    charge_doub_1rt = Path(hist_double / str('charge_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                             '_double_1x_rt.txt'))
    charge_doub_15rt = Path(hist_double / str('charge_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                              '_double_1.5x_rt.txt'))
    charge_doub_2rt = Path(hist_double / str('charge_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                             '_double_2x_rt.txt'))
    charge_doub_25rt = Path(hist_double / str('charge_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                              '_double_2.5x_rt.txt'))
    charge_doub_3rt = Path(hist_double / str('charge_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                             '_double_3x_rt.txt'))
    charge_doub_35rt = Path(hist_double / str('charge_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                              '_double_3.5x_rt.txt'))
    charge_doub_4rt = Path(hist_double / str('charge_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                             '_double_4x_rt.txt'))
    charge_doub_45rt = Path(hist_double / str('charge_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                              '_double_4.5x_rt.txt'))
    charge_doub_5rt = Path(hist_double / str('charge_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                             '_double_5x_rt.txt'))
    charge_doub_55rt = Path(hist_double / str('charge_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                              '_double_5.5x_rt.txt'))
    charge_doub_6rt = Path(hist_double / str('charge_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                             '_double_6x_rt.txt'))
    charge_doub_40ns = Path(hist_double / str('charge_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                              '_double_40_ns.txt'))
    charge_doub_80ns = Path(hist_double / str('charge_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                              '_double_80_ns.txt'))

    fwhm_doub_no_delay = Path(hist_double / str('fwhm_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                                '_double_no_delay.txt'))
    fwhm_doub_05rt = Path(hist_double / str('fwhm_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                            '_double_0.5x_rt.txt'))
    fwhm_doub_1rt = Path(hist_double / str('fwhm_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                           '_double_1x_rt.txt'))
    fwhm_doub_15rt = Path(hist_double / str('fwhm_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                            '_double_1.5x_rt.txt'))
    fwhm_doub_2rt = Path(hist_double / str('fwhm_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                           '_double_2x_rt.txt'))
    fwhm_doub_25rt = Path(hist_double / str('fwhm_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                            '_double_2.5x_rt.txt'))
    fwhm_doub_3rt = Path(hist_double / str('fwhm_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                           '_double_3x_rt.txt'))
    fwhm_doub_35rt = Path(hist_double / str('fwhm_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                            '_double_3.5x_rt.txt'))
    fwhm_doub_4rt = Path(hist_double / str('fwhm_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                           '_double_4x_rt.txt'))
    fwhm_doub_45rt = Path(hist_double / str('fwhm_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                            '_double_4.5x_rt.txt'))
    fwhm_doub_5rt = Path(hist_double / str('fwhm_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                           '_double_5x_rt.txt'))
    fwhm_doub_55rt = Path(hist_double / str('fwhm_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                            '_double_5.5x_rt.txt'))
    fwhm_doub_6rt = Path(hist_double / str('fwhm_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                           '_double_6x_rt.txt'))
    fwhm_doub_40ns = Path(hist_double / str('fwhm_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                            '_double_40_ns.txt'))
    fwhm_doub_80ns = Path(hist_double / str('fwhm_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                            '_double_80_ns.txt'))

    return amp_sing, charge_sing, fwhm_sing, amp_doub_no_delay, amp_doub_05rt, amp_doub_1rt, amp_doub_15rt, \
           amp_doub_2rt, amp_doub_25rt, amp_doub_3rt, amp_doub_35rt, amp_doub_4rt, amp_doub_45rt, amp_doub_5rt, \
           amp_doub_55rt, amp_doub_6rt, amp_doub_40ns, amp_doub_80ns, charge_doub_no_delay, charge_doub_05rt, \
           charge_doub_1rt, charge_doub_15rt, charge_doub_2rt, charge_doub_25rt, charge_doub_3rt, charge_doub_35rt, \
           charge_doub_4rt, charge_doub_45rt, charge_doub_5rt, charge_doub_55rt, charge_doub_6rt, charge_doub_40ns, \
           charge_doub_80ns, fwhm_doub_no_delay, fwhm_doub_05rt, fwhm_doub_1rt, fwhm_doub_15rt, fwhm_doub_2rt, \
           fwhm_doub_25rt, fwhm_doub_3rt, fwhm_doub_35rt, fwhm_doub_4rt, fwhm_doub_45rt, fwhm_doub_5rt, fwhm_doub_55rt,\
           fwhm_doub_6rt, fwhm_doub_40ns, fwhm_doub_80ns


# Creates arrays of data in histogram files and calculates mean and SD
def hist_data(amp_sing, charge_sing, fwhm_sing, amp_doub_no_delay, amp_doub_05rt, amp_doub_1rt, amp_doub_15rt,
                amp_doub_2rt, amp_doub_25rt, amp_doub_3rt, amp_doub_35rt, amp_doub_4rt, amp_doub_45rt, amp_doub_5rt,
                amp_doub_55rt, amp_doub_6rt, amp_doub_40ns, amp_doub_80ns, charge_doub_no_delay, charge_doub_05rt,
                charge_doub_1rt, charge_doub_15rt, charge_doub_2rt, charge_doub_25rt, charge_doub_3rt, charge_doub_35rt,
                charge_doub_4rt, charge_doub_45rt, charge_doub_5rt, charge_doub_55rt, charge_doub_6rt, charge_doub_40ns,
                charge_doub_80ns, fwhm_doub_no_delay, fwhm_doub_05rt, fwhm_doub_1rt, fwhm_doub_15rt, fwhm_doub_2rt,
                fwhm_doub_25rt, fwhm_doub_3rt, fwhm_doub_35rt, fwhm_doub_4rt, fwhm_doub_45rt, fwhm_doub_5rt,
                fwhm_doub_55rt, fwhm_doub_6rt, fwhm_doub_40ns, fwhm_doub_80ns, fsps_new):
    amp_sing_array = read_hist_file(amp_sing, fsps_new)
    charge_sing_array = read_hist_file(charge_sing, fsps_new)
    fwhm_sing_array = read_hist_file(fwhm_sing, fsps_new)
    amp_doub_no_delay_array = read_hist_file(amp_doub_no_delay, fsps_new)
    amp_doub_05rt_array = read_hist_file(amp_doub_05rt, fsps_new)
    amp_doub_1rt_array = read_hist_file(amp_doub_1rt, fsps_new)
    amp_doub_15rt_array = read_hist_file(amp_doub_15rt, fsps_new)
    amp_doub_2rt_array = read_hist_file(amp_doub_2rt, fsps_new)
    amp_doub_25rt_array = read_hist_file(amp_doub_25rt, fsps_new)
    amp_doub_3rt_array = read_hist_file(amp_doub_3rt, fsps_new)
    amp_doub_35rt_array = read_hist_file(amp_doub_35rt, fsps_new)
    amp_doub_4rt_array = read_hist_file(amp_doub_4rt, fsps_new)
    amp_doub_45rt_array = read_hist_file(amp_doub_45rt, fsps_new)
    amp_doub_5rt_array = read_hist_file(amp_doub_5rt, fsps_new)
    amp_doub_55rt_array = read_hist_file(amp_doub_55rt, fsps_new)
    amp_doub_6rt_array = read_hist_file(amp_doub_6rt, fsps_new)
    amp_doub_40ns_array = read_hist_file(amp_doub_40ns, fsps_new)
    amp_doub_80ns_array = read_hist_file(amp_doub_80ns, fsps_new)
    charge_doub_no_delay_array = read_hist_file(charge_doub_no_delay, fsps_new)
    charge_doub_05rt_array = read_hist_file(charge_doub_05rt, fsps_new)
    charge_doub_1rt_array = read_hist_file(charge_doub_1rt, fsps_new)
    charge_doub_15rt_array = read_hist_file(charge_doub_15rt, fsps_new)
    charge_doub_2rt_array = read_hist_file(charge_doub_2rt, fsps_new)
    charge_doub_25rt_array = read_hist_file(charge_doub_25rt, fsps_new)
    charge_doub_3rt_array = read_hist_file(charge_doub_3rt, fsps_new)
    charge_doub_35rt_array = read_hist_file(charge_doub_35rt, fsps_new)
    charge_doub_4rt_array = read_hist_file(charge_doub_4rt, fsps_new)
    charge_doub_45rt_array = read_hist_file(charge_doub_45rt, fsps_new)
    charge_doub_5rt_array = read_hist_file(charge_doub_5rt, fsps_new)
    charge_doub_55rt_array = read_hist_file(charge_doub_55rt, fsps_new)
    charge_doub_6rt_array = read_hist_file(charge_doub_6rt, fsps_new)
    charge_doub_40ns_array = read_hist_file(charge_doub_40ns, fsps_new)
    charge_doub_80ns_array = read_hist_file(charge_doub_80ns, fsps_new)
    fwhm_doub_no_delay_array = read_hist_file(fwhm_doub_no_delay, fsps_new)
    fwhm_doub_05rt_array = read_hist_file(fwhm_doub_05rt, fsps_new)
    fwhm_doub_1rt_array = read_hist_file(fwhm_doub_1rt, fsps_new)
    fwhm_doub_15rt_array = read_hist_file(fwhm_doub_15rt, fsps_new)
    fwhm_doub_2rt_array = read_hist_file(fwhm_doub_2rt, fsps_new)
    fwhm_doub_25rt_array = read_hist_file(fwhm_doub_25rt, fsps_new)
    fwhm_doub_3rt_array = read_hist_file(fwhm_doub_3rt, fsps_new)
    fwhm_doub_35rt_array = read_hist_file(fwhm_doub_35rt, fsps_new)
    fwhm_doub_4rt_array = read_hist_file(fwhm_doub_4rt, fsps_new)
    fwhm_doub_45rt_array = read_hist_file(fwhm_doub_45rt, fsps_new)
    fwhm_doub_5rt_array = read_hist_file(fwhm_doub_5rt, fsps_new)
    fwhm_doub_55rt_array = read_hist_file(fwhm_doub_55rt, fsps_new)
    fwhm_doub_6rt_array = read_hist_file(fwhm_doub_6rt, fsps_new)
    fwhm_doub_40ns_array = read_hist_file(fwhm_doub_40ns, fsps_new)
    fwhm_doub_80ns_array = read_hist_file(fwhm_doub_80ns, fsps_new)

    amp_sing_mean = np.mean(amp_sing_array)
    charge_sing_mean = np.mean(charge_sing_array)
    fwhm_sing_mean = np.mean(fwhm_sing_array)
    amp_doub_no_delay_mean = np.mean(amp_doub_no_delay_array)
    amp_doub_05rt_mean = np.mean(amp_doub_05rt_array)
    amp_doub_1rt_mean = np.mean(amp_doub_1rt_array)
    amp_doub_15rt_mean = np.mean(amp_doub_15rt_array)
    amp_doub_2rt_mean = np.mean(amp_doub_2rt_array)
    amp_doub_25rt_mean = np.mean(amp_doub_25rt_array)
    amp_doub_3rt_mean = np.mean(amp_doub_3rt_array)
    amp_doub_35rt_mean = np.mean(amp_doub_35rt_array)
    amp_doub_4rt_mean = np.mean(amp_doub_4rt_array)
    amp_doub_45rt_mean = np.mean(amp_doub_45rt_array)
    amp_doub_5rt_mean = np.mean(amp_doub_5rt_array)
    amp_doub_55rt_mean = np.mean(amp_doub_55rt_array)
    amp_doub_6rt_mean = np.mean(amp_doub_6rt_array)
    amp_doub_40ns_mean = np.mean(amp_doub_40ns_array)
    amp_doub_80ns_mean = np.mean(amp_doub_80ns_array)
    charge_doub_no_delay_mean = np.mean(charge_doub_no_delay_array)
    charge_doub_05rt_mean = np.mean(charge_doub_05rt_array)
    charge_doub_1rt_mean = np.mean(charge_doub_1rt_array)
    charge_doub_15rt_mean = np.mean(charge_doub_15rt_array)
    charge_doub_2rt_mean = np.mean(charge_doub_2rt_array)
    charge_doub_25rt_mean = np.mean(charge_doub_25rt_array)
    charge_doub_3rt_mean = np.mean(charge_doub_3rt_array)
    charge_doub_35rt_mean = np.mean(charge_doub_35rt_array)
    charge_doub_4rt_mean = np.mean(charge_doub_4rt_array)
    charge_doub_45rt_mean = np.mean(charge_doub_45rt_array)
    charge_doub_5rt_mean = np.mean(charge_doub_5rt_array)
    charge_doub_55rt_mean = np.mean(charge_doub_55rt_array)
    charge_doub_6rt_mean = np.mean(charge_doub_6rt_array)
    charge_doub_40ns_mean = np.mean(charge_doub_40ns_array)
    charge_doub_80ns_mean = np.mean(charge_doub_80ns_array)
    fwhm_doub_no_delay_mean = np.mean(fwhm_doub_no_delay_array)
    fwhm_doub_05rt_mean = np.mean(fwhm_doub_05rt_array)
    fwhm_doub_1rt_mean = np.mean(fwhm_doub_1rt_array)
    fwhm_doub_15rt_mean = np.mean(fwhm_doub_15rt_array)
    fwhm_doub_2rt_mean = np.mean(fwhm_doub_2rt_array)
    fwhm_doub_25rt_mean = np.mean(fwhm_doub_25rt_array)
    fwhm_doub_3rt_mean = np.mean(fwhm_doub_3rt_array)
    fwhm_doub_35rt_mean = np.mean(fwhm_doub_35rt_array)
    fwhm_doub_4rt_mean = np.mean(fwhm_doub_4rt_array)
    fwhm_doub_45rt_mean = np.mean(fwhm_doub_45rt_array)
    fwhm_doub_5rt_mean = np.mean(fwhm_doub_5rt_array)
    fwhm_doub_55rt_mean = np.mean(fwhm_doub_55rt_array)
    fwhm_doub_6rt_mean = np.mean(fwhm_doub_6rt_array)
    fwhm_doub_40ns_mean = np.mean(fwhm_doub_40ns_array)
    fwhm_doub_80ns_mean = np.mean(fwhm_doub_80ns_array)

    amp_sing_std = np.std(amp_sing_array)
    charge_sing_std = np.std(charge_sing_array)
    fwhm_sing_std = np.std(fwhm_sing_array)
    amp_doub_no_delay_std = np.std(amp_doub_no_delay_array)
    amp_doub_05rt_std = np.std(amp_doub_05rt_array)
    amp_doub_1rt_std = np.std(amp_doub_1rt_array)
    amp_doub_15rt_std = np.std(amp_doub_15rt_array)
    amp_doub_2rt_std = np.std(amp_doub_2rt_array)
    amp_doub_25rt_std = np.std(amp_doub_25rt_array)
    amp_doub_3rt_std = np.std(amp_doub_3rt_array)
    amp_doub_35rt_std = np.std(amp_doub_35rt_array)
    amp_doub_4rt_std = np.std(amp_doub_4rt_array)
    amp_doub_45rt_std = np.std(amp_doub_45rt_array)
    amp_doub_5rt_std = np.std(amp_doub_5rt_array)
    amp_doub_55rt_std = np.std(amp_doub_55rt_array)
    amp_doub_6rt_std = np.std(amp_doub_6rt_array)
    amp_doub_40ns_std = np.std(amp_doub_40ns_array)
    amp_doub_80ns_std = np.std(amp_doub_80ns_array)
    charge_doub_no_delay_std = np.std(charge_doub_no_delay_array)
    charge_doub_05rt_std = np.std(charge_doub_05rt_array)
    charge_doub_1rt_std = np.std(charge_doub_1rt_array)
    charge_doub_15rt_std = np.std(charge_doub_15rt_array)
    charge_doub_2rt_std = np.std(charge_doub_2rt_array)
    charge_doub_25rt_std = np.std(charge_doub_25rt_array)
    charge_doub_3rt_std = np.std(charge_doub_3rt_array)
    charge_doub_35rt_std = np.std(charge_doub_35rt_array)
    charge_doub_4rt_std = np.std(charge_doub_4rt_array)
    charge_doub_45rt_std = np.std(charge_doub_45rt_array)
    charge_doub_5rt_std = np.std(charge_doub_5rt_array)
    charge_doub_55rt_std = np.std(charge_doub_55rt_array)
    charge_doub_6rt_std = np.std(charge_doub_6rt_array)
    charge_doub_40ns_std = np.std(charge_doub_40ns_array)
    charge_doub_80ns_std = np.std(charge_doub_80ns_array)
    fwhm_doub_no_delay_std = np.std(fwhm_doub_no_delay_array)
    fwhm_doub_05rt_std = np.std(fwhm_doub_05rt_array)
    fwhm_doub_1rt_std = np.std(fwhm_doub_1rt_array)
    fwhm_doub_15rt_std = np.std(fwhm_doub_15rt_array)
    fwhm_doub_2rt_std = np.std(fwhm_doub_2rt_array)
    fwhm_doub_25rt_std = np.std(fwhm_doub_25rt_array)
    fwhm_doub_3rt_std = np.std(fwhm_doub_3rt_array)
    fwhm_doub_35rt_std = np.std(fwhm_doub_35rt_array)
    fwhm_doub_4rt_std = np.std(fwhm_doub_4rt_array)
    fwhm_doub_45rt_std = np.std(fwhm_doub_45rt_array)
    fwhm_doub_5rt_std = np.std(fwhm_doub_5rt_array)
    fwhm_doub_55rt_std = np.std(fwhm_doub_55rt_array)
    fwhm_doub_6rt_std = np.std(fwhm_doub_6rt_array)
    fwhm_doub_40ns_std = np.std(fwhm_doub_40ns_array)
    fwhm_doub_80ns_std = np.std(fwhm_doub_80ns_array)

    return amp_sing_mean, charge_sing_mean, fwhm_sing_mean, amp_doub_no_delay_mean, amp_doub_05rt_mean, \
           amp_doub_1rt_mean, amp_doub_15rt_mean, amp_doub_2rt_mean, amp_doub_25rt_mean, amp_doub_3rt_mean, \
           amp_doub_35rt_mean, amp_doub_4rt_mean, amp_doub_45rt_mean, amp_doub_5rt_mean, amp_doub_55rt_mean, \
           amp_doub_6rt_mean, amp_doub_40ns_mean, amp_doub_80ns_mean, charge_doub_no_delay_mean, charge_doub_05rt_mean,\
           charge_doub_1rt_mean, charge_doub_15rt_mean, charge_doub_2rt_mean, charge_doub_25rt_mean, \
           charge_doub_3rt_mean, charge_doub_35rt_mean, charge_doub_4rt_mean, charge_doub_45rt_mean, \
           charge_doub_5rt_mean, charge_doub_55rt_mean, charge_doub_6rt_mean, charge_doub_40ns_mean, \
           charge_doub_80ns_mean, fwhm_doub_no_delay_mean, fwhm_doub_05rt_mean, fwhm_doub_1rt_mean, \
           fwhm_doub_15rt_mean, fwhm_doub_2rt_mean, fwhm_doub_25rt_mean, fwhm_doub_3rt_mean, fwhm_doub_35rt_mean, \
           fwhm_doub_4rt_mean, fwhm_doub_45rt_mean, fwhm_doub_5rt_mean, fwhm_doub_55rt_mean, fwhm_doub_6rt_mean, \
           fwhm_doub_40ns_mean, fwhm_doub_80ns_mean, amp_sing_std, charge_sing_std, fwhm_sing_std, \
           amp_doub_no_delay_std, amp_doub_05rt_std, amp_doub_1rt_std, amp_doub_15rt_std, amp_doub_2rt_std, \
           amp_doub_25rt_std, amp_doub_3rt_std, amp_doub_35rt_std, amp_doub_4rt_std, amp_doub_45rt_std, \
           amp_doub_5rt_std, amp_doub_55rt_std, amp_doub_6rt_std, amp_doub_40ns_std, amp_doub_80ns_std, \
           charge_doub_no_delay_std, charge_doub_05rt_std, charge_doub_1rt_std, charge_doub_15rt_std, \
           charge_doub_2rt_std, charge_doub_25rt_std, charge_doub_3rt_std, charge_doub_35rt_std, charge_doub_4rt_std, \
           charge_doub_45rt_std, charge_doub_5rt_std, charge_doub_55rt_std, charge_doub_6rt_std, charge_doub_40ns_std, \
           charge_doub_80ns_std, fwhm_doub_no_delay_std, fwhm_doub_05rt_std, fwhm_doub_1rt_std, fwhm_doub_15rt_std, \
           fwhm_doub_2rt_std, fwhm_doub_25rt_std, fwhm_doub_3rt_std, fwhm_doub_35rt_std, fwhm_doub_4rt_std, \
           fwhm_doub_45rt_std, fwhm_doub_5rt_std, fwhm_doub_55rt_std, fwhm_doub_6rt_std, fwhm_doub_40ns_std, \
           fwhm_doub_80ns_std


# Creates plots of false SPE rate vs delay
def false_spes_vs_delay(start, end, factor, parameter, parameter_title, units, fsps_new, means, meanno, mean05, mean1,
                        mean15, mean2, mean25, mean3, mean35, mean4, mean45, mean5, mean55, mean6, mean40, mean80,
                        sds, sdno, sd05, sd1, sd15, sd2, sd25, sd3, sd35, sd4, sd45, sd5, sd55, sd6, sd40, sd80,
                        dest_path, shaping):
    cutoff_array = np.array([])
    spes_as_mpes_array = np.array([])
    mpes_as_spes_array = np.array([])
    mpes_as_spes_no_array = np.array([])
    mpes_as_spes_05x_array = np.array([])
    mpes_as_spes_1x_array = np.array([])
    mpes_as_spes_15x_array = np.array([])
    mpes_as_spes_2x_array = np.array([])
    mpes_as_spes_25x_array = np.array([])
    mpes_as_spes_3x_array = np.array([])
    mpes_as_spes_35x_array = np.array([])
    mpes_as_spes_4x_array = np.array([])
    mpes_as_spes_45x_array = np.array([])
    mpes_as_spes_5x_array = np.array([])
    mpes_as_spes_55x_array = np.array([])
    mpes_as_spes_6x_array = np.array([])
    mpes_as_spes_40_array = np.array([])
    mpes_as_spes_80_array = np.array([])

    for i in range(start, end):
        x = i * factor
        cutoff_array = np.append(cutoff_array, x)
        spes_as_mpes = 100 * (1 + ((1 / 2) * (-2 + math.erfc((x - means) / (sds * math.sqrt(2))))))
        mpes_as_spes_no = 100 * ((1 / 2) * (2 - math.erfc((x - meanno) / (sdno * math.sqrt(2)))))
        mpes_as_spes_05x = 100 * ((1 / 2) * (2 - math.erfc((x - mean05) / (sd05 * math.sqrt(2)))))
        mpes_as_spes_1x = 100 * ((1 / 2) * (2 - math.erfc((x - mean1) / (sd1 * math.sqrt(2)))))
        mpes_as_spes_15x = 100 * ((1 / 2) * (2 - math.erfc((x - mean15) / (sd15 * math.sqrt(2)))))
        mpes_as_spes_2x = 100 * ((1 / 2) * (2 - math.erfc((x - mean2) / (sd2 * math.sqrt(2)))))
        mpes_as_spes_25x = 100 * ((1 / 2) * (2 - math.erfc((x - mean25) / (sd25 * math.sqrt(2)))))
        mpes_as_spes_3x = 100 * ((1 / 2) * (2 - math.erfc((x - mean3) / (sd3 * math.sqrt(2)))))
        mpes_as_spes_35x = 100 * ((1 / 2) * (2 - math.erfc((x - mean35) / (sd35 * math.sqrt(2)))))
        mpes_as_spes_4x = 100 * ((1 / 2) * (2 - math.erfc((x - mean4) / (sd4 * math.sqrt(2)))))
        mpes_as_spes_45x = 100 * ((1 / 2) * (2 - math.erfc((x - mean45) / (sd45 * math.sqrt(2)))))
        mpes_as_spes_5x = 100 * ((1 / 2) * (2 - math.erfc((x - mean5) / (sd5 * math.sqrt(2)))))
        mpes_as_spes_55x = 100 * ((1 / 2) * (2 - math.erfc((x - mean55) / (sd55 * math.sqrt(2)))))
        mpes_as_spes_6x = 100 * ((1 / 2) * (2 - math.erfc((x - mean6) / (sd6 * math.sqrt(2)))))
        mpes_as_spes_40 = 100 * ((1 / 2) * (2 - math.erfc((x - mean40) / (sd40 * math.sqrt(2)))))
        mpes_as_spes_80 = 100 * ((1 / 2) * (2 - math.erfc((x - mean80) / (sd80 * math.sqrt(2)))))
        spes_as_mpes_array = np.append(spes_as_mpes_array, spes_as_mpes)
        mpes_as_spes_no_array = np.append(mpes_as_spes_no_array, mpes_as_spes_no)
        mpes_as_spes_05x_array = np.append(mpes_as_spes_05x_array, mpes_as_spes_05x)
        mpes_as_spes_1x_array = np.append(mpes_as_spes_1x_array, mpes_as_spes_1x)
        mpes_as_spes_15x_array = np.append(mpes_as_spes_15x_array, mpes_as_spes_15x)
        mpes_as_spes_2x_array = np.append(mpes_as_spes_2x_array, mpes_as_spes_2x)
        mpes_as_spes_25x_array = np.append(mpes_as_spes_25x_array, mpes_as_spes_25x)
        mpes_as_spes_3x_array = np.append(mpes_as_spes_3x_array, mpes_as_spes_3x)
        mpes_as_spes_35x_array = np.append(mpes_as_spes_35x_array, mpes_as_spes_35x)
        mpes_as_spes_4x_array = np.append(mpes_as_spes_4x_array, mpes_as_spes_4x)
        mpes_as_spes_45x_array = np.append(mpes_as_spes_45x_array, mpes_as_spes_45x)
        mpes_as_spes_5x_array = np.append(mpes_as_spes_5x_array, mpes_as_spes_5x)
        mpes_as_spes_55x_array = np.append(mpes_as_spes_55x_array, mpes_as_spes_55x)
        mpes_as_spes_6x_array = np.append(mpes_as_spes_6x_array, mpes_as_spes_6x)
        mpes_as_spes_40_array = np.append(mpes_as_spes_40_array, mpes_as_spes_40)
        mpes_as_spes_80_array = np.append(mpes_as_spes_80_array, mpes_as_spes_80)

    cutoff_array_2 = np.linspace(start * factor, end * factor, 1000)
    spes_as_mpes_array_2 = np.interp(cutoff_array_2, cutoff_array, spes_as_mpes_array)
    mpes_as_spes_no_array_2 = np.interp(cutoff_array_2, cutoff_array, mpes_as_spes_no_array)
    mpes_as_spes_05x_array_2 = np.interp(cutoff_array_2, cutoff_array, mpes_as_spes_05x_array)
    mpes_as_spes_1x_array_2 = np.interp(cutoff_array_2, cutoff_array, mpes_as_spes_1x_array)
    mpes_as_spes_15x_array_2 = np.interp(cutoff_array_2, cutoff_array, mpes_as_spes_15x_array)
    mpes_as_spes_2x_array_2 = np.interp(cutoff_array_2, cutoff_array, mpes_as_spes_2x_array)
    mpes_as_spes_25x_array_2 = np.interp(cutoff_array_2, cutoff_array, mpes_as_spes_25x_array)
    mpes_as_spes_3x_array_2 = np.interp(cutoff_array_2, cutoff_array, mpes_as_spes_3x_array)
    mpes_as_spes_35x_array_2 = np.interp(cutoff_array_2, cutoff_array, mpes_as_spes_35x_array)
    mpes_as_spes_4x_array_2 = np.interp(cutoff_array_2, cutoff_array, mpes_as_spes_4x_array)
    mpes_as_spes_45x_array_2 = np.interp(cutoff_array_2, cutoff_array, mpes_as_spes_45x_array)
    mpes_as_spes_5x_array_2 = np.interp(cutoff_array_2, cutoff_array, mpes_as_spes_5x_array)
    mpes_as_spes_55x_array_2 = np.interp(cutoff_array_2, cutoff_array, mpes_as_spes_55x_array)
    mpes_as_spes_6x_array_2 = np.interp(cutoff_array_2, cutoff_array, mpes_as_spes_6x_array)
    mpes_as_spes_40_array_2 = np.interp(cutoff_array_2, cutoff_array, mpes_as_spes_40_array)
    mpes_as_spes_80_array_2 = np.interp(cutoff_array_2, cutoff_array, mpes_as_spes_80_array)

    idx = np.argmin(np.abs(spes_as_mpes_array_2 - 1))
    mpes_as_spes_array = np.append(mpes_as_spes_array, mpes_as_spes_no_array_2[idx])
    mpes_as_spes_array = np.append(mpes_as_spes_array, mpes_as_spes_05x_array_2[idx])
    mpes_as_spes_array = np.append(mpes_as_spes_array, mpes_as_spes_1x_array_2[idx])
    mpes_as_spes_array = np.append(mpes_as_spes_array, mpes_as_spes_15x_array_2[idx])
    mpes_as_spes_array = np.append(mpes_as_spes_array, mpes_as_spes_2x_array_2[idx])
    mpes_as_spes_array = np.append(mpes_as_spes_array, mpes_as_spes_25x_array_2[idx])
    mpes_as_spes_array = np.append(mpes_as_spes_array, mpes_as_spes_3x_array_2[idx])
    mpes_as_spes_array = np.append(mpes_as_spes_array, mpes_as_spes_35x_array_2[idx])
    mpes_as_spes_array = np.append(mpes_as_spes_array, mpes_as_spes_4x_array_2[idx])
    mpes_as_spes_array = np.append(mpes_as_spes_array, mpes_as_spes_45x_array_2[idx])
    mpes_as_spes_array = np.append(mpes_as_spes_array, mpes_as_spes_5x_array_2[idx])
    mpes_as_spes_array = np.append(mpes_as_spes_array, mpes_as_spes_55x_array_2[idx])
    mpes_as_spes_array = np.append(mpes_as_spes_array, mpes_as_spes_6x_array_2[idx])
    mpes_as_spes_array = np.append(mpes_as_spes_array, mpes_as_spes_40_array_2[idx])
    mpes_as_spes_array = np.append(mpes_as_spes_array, mpes_as_spes_80_array_2[idx])

    delay_array = np.array([0, 1.52e-9, 3.04e-9, 4.56e-9, 6.08e-9, 7.6e-9, 9.12e-9, 1.064e-8, 1.216e-8, 1.368e-8,
                            1.52e-8, 1.672e-8, 1.824e-8, 4e-8, 8e-8])
    cutoff = str(float(format(cutoff_array_2[idx], '.2e')))

    plt.scatter(delay_array, mpes_as_spes_array)
    plt.plot(delay_array, mpes_as_spes_array)
    plt.xlim(-1e-8, 8.2e-8)
    plt.ylim(-5, 100)
    plt.xlabel('Delay (s)')
    plt.ylabel('% MPES Misidentified as SPEs')
    plt.title('False SPEs\n' + parameter_title + ' Cutoff = ' + cutoff + ' (' + units + ') (1% False MPEs)')
    for i in range(len(mpes_as_spes_array)):
        pt = str(float(format(mpes_as_spes_array[i], '.1e')))
        plt.annotate(pt + '%', (delay_array[i], mpes_as_spes_array[i] + 1))
    plt.savefig(dest_path / 'plots' / str('false_spes_delay_' + parameter + '_' + str(int(fsps_new / 1e6)) + '_Msps_' +
                                          shaping + '.png'), dpi=360)
    plt.close()


# Creates plots of false SPE & MPE rate vs cutoff
def false_spes_mpes(start, end, factor, parameter, parameter_title, units, means, meand, sds, sdd, fsps_new, dest_path,
                    shaping):
    cutoff_array = np.array([])
    spes_as_mpes_array = np.array([])
    mpes_as_spes_array = np.array([])

    for i in range(start, end):
        x = i * factor
        cutoff_array = np.append(cutoff_array, x)
        spes_as_mpes = 100 * (1 + ((1 / 2) * (-2 + math.erfc((x - means) / (sds * math.sqrt(2))))))
        mpes_as_spes = 100 * ((1 / 2) * (2 - math.erfc((x - meand) / (sdd * math.sqrt(2)))))
        spes_as_mpes_array = np.append(spes_as_mpes_array, spes_as_mpes)
        mpes_as_spes_array = np.append(mpes_as_spes_array, mpes_as_spes)

    cutoff_array_2 = np.linspace(start * factor, end * factor, 1000)
    spes_as_mpes_array_2 = np.interp(cutoff_array_2, cutoff_array, spes_as_mpes_array)
    mpes_as_spes_array_2 = np.interp(cutoff_array_2, cutoff_array, mpes_as_spes_array)
    idx = np.argmin(np.abs(spes_as_mpes_array_2 - 1))
    cutoff = float(format(cutoff_array_2[idx], '.2e'))
    false_spe = float(format(mpes_as_spes_array_2[idx], '.2e'))

    plt.plot(cutoff_array, spes_as_mpes_array)
    plt.ylim(-5, 100)
    plt.hlines(1, start * factor, end * factor)
    plt.xlabel(parameter_title + ' Cutoff (' + units + ')')
    plt.ylabel('% SPES Misidentified as MPEs')
    plt.title('False MPEs\n' + parameter_title + ' Cutoff: ' + str(cutoff) + ' ' + units)
    plt.annotate('1% false MPEs', (start * factor, 3))
    plt.savefig(dest_path / 'plots' / str('false_mpes_' + parameter + '_' + str(int(fsps_new / 1e6)) + '_Msps_' +
                                          shaping + '.png'), dpi=360)
    plt.close()

    plt.plot(cutoff_array, mpes_as_spes_array)
    plt.ylim(-5, 100)
    plt.vlines(cutoff, 0, 100)
    plt.xlabel(parameter_title + ' Cutoff (' + units + ')')
    plt.ylabel('% MPES Misidentified as SPEs')
    plt.title('False SPEs\n' + parameter_title + ' Cutoff: ' + str(cutoff) + ' ' + units)
    plt.annotate(str(str(false_spe) + '% false SPEs'), (cutoff, -2))
    plt.savefig(dest_path / 'plots' / str('false_spes_' + parameter + '_' + str(int(fsps_new / 1e6)) + '_Msps_' +
                                          shaping + '.png'), dpi=360)
    plt.close()


# Creates plots displaying cutoffs and false SPE rates
def make_plots(amp_sing_mean, charge_sing_mean, fwhm_sing_mean, amp_doub_no_delay_mean, charge_doub_no_delay_mean,
               fwhm_doub_no_delay_mean, fwhm_doub_05rt_mean, fwhm_doub_1rt_mean, fwhm_doub_15rt_mean,
               fwhm_doub_2rt_mean, fwhm_doub_25rt_mean, fwhm_doub_3rt_mean, fwhm_doub_35rt_mean, fwhm_doub_4rt_mean,
               fwhm_doub_45rt_mean, fwhm_doub_5rt_mean, fwhm_doub_55rt_mean, fwhm_doub_6rt_mean, fwhm_doub_40ns_mean,
               fwhm_doub_80ns_mean, amp_sing_std, charge_sing_std, fwhm_sing_std, amp_doub_no_delay_std,
               charge_doub_no_delay_std, fwhm_doub_no_delay_std, fwhm_doub_05rt_std, fwhm_doub_1rt_std,
               fwhm_doub_15rt_std, fwhm_doub_2rt_std, fwhm_doub_25rt_std, fwhm_doub_3rt_std, fwhm_doub_35rt_std,
               fwhm_doub_4rt_std, fwhm_doub_45rt_std, fwhm_doub_5rt_std, fwhm_doub_55rt_std, fwhm_doub_6rt_std,
               fwhm_doub_40ns_std, fwhm_doub_80ns_std, shaping, fsps_new, dest_path):
    print('Making plots...')

    if shaping == 'rt_1':
        start_fwhm = 5
        end_fwhm = 35
        factor_fwhm = 10 ** -9
        start_charge = 10
        end_charge = 125
        factor_charge = 10 ** -9
        start_amp = 50
        end_amp = 550
        factor_amp = 1
    elif shaping == 'rt_2':
        start_fwhm = 10
        end_fwhm = 60
        factor_fwhm = 10 ** -9
        start_charge = 10
        end_charge = 125
        factor_charge = 10 ** -9
        start_amp = 25
        end_amp = 200
        factor_amp = 1
    else:
        start_fwhm = 20
        end_fwhm = 80
        factor_fwhm = 10 ** -9
        start_charge = 10
        end_charge = 125
        factor_charge = 10 ** -9
        start_amp = 20
        end_amp = 150
        factor_amp = 1

    false_spes_vs_delay(start_fwhm, end_fwhm, factor_fwhm, 'fwhm', 'FWHM', 's', fsps_new, fwhm_sing_mean,
                        fwhm_doub_no_delay_mean, fwhm_doub_05rt_mean, fwhm_doub_1rt_mean, fwhm_doub_15rt_mean,
                        fwhm_doub_2rt_mean, fwhm_doub_25rt_mean, fwhm_doub_3rt_mean, fwhm_doub_35rt_mean,
                        fwhm_doub_4rt_mean, fwhm_doub_45rt_mean, fwhm_doub_5rt_mean, fwhm_doub_55rt_mean,
                        fwhm_doub_6rt_mean, fwhm_doub_40ns_mean, fwhm_doub_80ns_mean, fwhm_sing_std,
                        fwhm_doub_no_delay_std, fwhm_doub_05rt_std, fwhm_doub_1rt_std, fwhm_doub_15rt_std,
                        fwhm_doub_2rt_std, fwhm_doub_25rt_std, fwhm_doub_3rt_std, fwhm_doub_35rt_std, fwhm_doub_4rt_std,
                        fwhm_doub_45rt_std, fwhm_doub_5rt_std, fwhm_doub_55rt_std, fwhm_doub_6rt_std,
                        fwhm_doub_40ns_std, fwhm_doub_80ns_std, dest_path, shaping)

    false_spes_mpes(start_charge, end_charge, factor_charge, 'charge', 'Charge', 's*bit/ohm', charge_sing_mean,
                    charge_doub_no_delay_mean, charge_sing_std, charge_doub_no_delay_std, fsps_new, dest_path, shaping)
    false_spes_mpes(start_amp, end_amp, factor_amp, 'amp', 'Amplitude', 'bits', amp_sing_mean, amp_doub_no_delay_mean,
                    amp_sing_std, amp_doub_no_delay_std, fsps_new, dest_path, shaping)
