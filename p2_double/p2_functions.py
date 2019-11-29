import os
import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy import signal


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
    file_name = str(dest_path / 'calculations' / 'D2--waveforms--%05d.txt') % i
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
    file_name = Path(dest_path / 'hist_data_double' / name)

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
def average_waveform(array, dest_path, shaping, shaping_name, delay_path, delay_name, delay_folder, nhdr):
    save_file = Path(dest_path / 'plots')
    tsum = 0
    vsum = 0
    n = 0

    for item in array:
        file_name = 'D2--waveforms--%05d.txt' % item
        if os.path.isfile(delay_path / file_name):
            print('Reading file #', item)
            t, v, hdr = rw(delay_path / file_name, nhdr)    # Reads a waveform file
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
    plt.title('Average Waveform (' + delay_name + ', ' + shaping_name + ')')
    plt.savefig(save_file / str('avg_waveform_double_' + delay_folder + "_" + shaping + '.png'), dpi=360)

    # Saves average waveform data
    file_name = dest_path / 'hist_data_double' / str('avg_waveform_' + delay_folder + "_" + shaping + '.txt')
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

    for i in range(len(v) - 1):
        if v[i] <= 0.1 * min(v):
            idx1 = i
            break
        else:
            continue

    if idx1 == np.inf:
        return 0, -1
    else:
        for i in range(idx1, len(v) - 1):
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
        for i in range(int(.05 * len(t)), int(np.where(t == t1)[0] - (.1 * len(t)))):
            v_sum += v[i]

        for i in range(int(np.where(t == t2)[0] + (.1 * len(t))), int(.95 * len(t))):
            v_sum += v[i]

        average = v_sum / ((int(np.where(t == t1)[0] - (.1 * len(t))) - int(.05 * len(t))) +
                           (int(.95 * len(t)) - int(np.where(t == t2)[0] + (.1 * len(t)))))

        return average

    except:
        return np.inf


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


# Puts voltage array through a lowpass filter given a tau and sample rate
def lowpass_filter(v, tau, fsps):
    v_filtered = np.array([])
    alpha = 1 - np.exp(-1. / (fsps * tau))
    for i in range(len(v)):
        if i == 0:
            v_filtered = np.append(v_filtered, v[i])
        else:
            v_filtered = np.append(v_filtered, v[i] * alpha + (1 - alpha) * v_filtered[i - 1])
    return v_filtered


# Calculates 10-90 rise time for each shaping and returns arrays of 10-90 rise times
def make_arrays(array, delay_path1, delay_path2, delay_path4, delay_path8, dest_path, nhdr):
    rt_1_array = np.array([])
    rt_2_array = np.array([])
    rt_4_array = np.array([])
    rt_8_array = np.array([])

    for item in array:
        file_name1 = str(delay_path1 / 'D2--waveforms--%05d.txt') % item
        file_name2 = str(delay_path2 / 'D2--waveforms--%05d.txt') % item
        file_name3 = str(delay_path4 / 'D2--waveforms--%05d.txt') % item
        file_name4 = str(delay_path8 / 'D2--waveforms--%05d.txt') % item
        file_name5 = str(dest_path / 'calculations_double' / 'D2--waveforms--%05d.txt') % item

        # If the calculations were done previously, they are read from a file
        if os.path.isfile(file_name5):
            print("Reading calculations from file #%05d" % i)
            risetime_1, risetime_2, risetime_4, risetime_8 = read_calc(file_name5)

            rt_1_array = np.append(rt_1_array, risetime_1)
            rt_2_array = np.append(rt_2_array, risetime_2)
            rt_4_array = np.append(rt_4_array, risetime_4)
            rt_8_array = np.append(rt_8_array, risetime_8)
        # If the calculations were not done yet, they are calculated
        else:
            if os.path.isfile(file_name1):
                print("Calculating file #%05d" % item)
                t1, v1, hdr = rw(file_name1, nhdr)          # Unshaped waveform file is read
                t2, v2, hdr = rw(file_name2, nhdr)          # 2x rise time waveform file is read
                t4, v4, hdr = rw(file_name3, nhdr)          # 4x rise time waveform file is read
                t8, v8, hdr = rw(file_name4, nhdr)          # 8x rise time waveform file is read
                risetime_1 = rise_time(t1, v1, 10, 90)      # Rise time calculation is done
                risetime_2 = rise_time(t2, v2, 10, 90)      # Rise time calculation is done
                risetime_4 = rise_time(t4, v4, 10, 90)      # Rise time calculation is done
                risetime_8 = rise_time(t8, v8, 10, 90)      # Rise time calculation is done
                save_calculations(dest_path, item, risetime_1, risetime_2, risetime_4, risetime_8)

                rt_1_array = np.append(rt_1_array, risetime_1)
                rt_2_array = np.append(rt_2_array, risetime_2)
                rt_4_array = np.append(rt_4_array, risetime_4)
                rt_8_array = np.append(rt_8_array, risetime_8)

    return rt_1_array, rt_2_array, rt_4_array, rt_8_array


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
    popt1, pcov1 = curve_fit(func, bins_range1, n_range1, p0=guess1, maxfev=10000)      # Finds Gaussian fit
    mu1 = float(format(popt1[1], '.2e'))                        # Calculates mean based on 1sigma guess
    sigma1 = np.abs(float(format(popt1[2], '.2e')))     # Calculates standard deviation based on 1sigma estimation
    range_min2 = mu1 - 2 * sigma1                       # Calculates lower limit of Gaussian fit (2sigma)
    range_max2 = mu1 + 2 * sigma1                       # Calculates upper limit of Gaussian fit (2sigma)
    bins_range2 = np.linspace(range_min2, range_max2, 10000)    # Creates array of bins between upper & lower limits
    n_range2 = np.interp(bins_range2, bins, n)          # Interpolates & creates array of y axis values
    guess2 = [1, mu1, sigma1]                           # Defines guess for values of a, b & c in Gaussian fit
    popt2, pcov2 = curve_fit(func, bins_range2, n_range2, p0=guess2, maxfev=10000)      # Finds Gaussian fit

    return bins_range2, popt2, pcov2


# Creates histogram given an array
def plot_histogram(array, dest_path, nbins, xaxis, title, units, filename):

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

    write_hist_data(array, dest_path, filename + '.txt')

    plt.close()


# Plots histograms for each calculation array
def p2_hist(rt_1_array, rt_2_array, rt_4_array, rt_8_array, dest_path, bins, delay_name, delay_folder):
    print('Creating histograms...')
    plot_histogram(rt_1_array, dest_path, bins, 'Time', '10-90 Rise Time (' + delay_name + ', No Shaping)', 's',
                   delay_folder + 'rt_1_double')
    plot_histogram(rt_2_array, dest_path, bins, 'Time', '10-90 Rise Time (' + delay_name + ', 2x Shaping)', 's',
                   delay_folder + 'rt_2_double')
    plot_histogram(rt_4_array, dest_path, bins, 'Time', '10-90 Rise Time (' + delay_name + ', 4x Shaping)', 's',
                   delay_folder + 'rt_4_double')
    plot_histogram(rt_8_array, dest_path, bins, 'Time', '10-90 Rise Time (' + delay_name + ', 8x Shaping)', 's',
                   delay_folder + 'rt_8_double')


# P2_CREATE_DOUBLE


# Creates p2 double folder names
def initialize_folders(date, filter_band, delay_folder):
    gen_path = Path(r'/Volumes/TOSHIBA EXT/data/watchman')
    save_path = Path(str(gen_path / '%08d_watchman_spe/waveforms/%s') % (date, filter_band))
    dest_path = Path(save_path / 'd2')
    single_path = Path(dest_path / 'rt_1_single')
    filt_path1 = Path(dest_path / 'rt_1_double')
    filt_path2 = Path(dest_path / 'rt_2_double')
    filt_path4 = Path(dest_path / 'rt_4_double')
    filt_path8 = Path(dest_path / 'rt_8_double')
    delay_path1 = Path(filt_path1 / delay_folder)
    delay_path2 = Path(filt_path2 / delay_folder)
    delay_path4 = Path(filt_path4 / delay_folder)
    delay_path8 = Path(filt_path8 / delay_folder)
    filt_path1_s = Path(dest_path / 'rt_1_single_2')
    filt_path2_s = Path(dest_path / 'rt_2_single_2')
    filt_path4_s = Path(dest_path / 'rt_4_single_2')
    filt_path8_s = Path(dest_path / 'rt_8_single_2')

    return gen_path, save_path, dest_path, single_path, filt_path1, filt_path2, filt_path4, filt_path8, delay_path1, \
           delay_path2, delay_path4, delay_path8, filt_path1_s, filt_path2_s, filt_path4_s, filt_path8_s


# Creates p2 double folders
def make_folders(dest_path, filt_path1, filt_path2, filt_path4, filt_path8, delay_path1, delay_path2, delay_path4,
                 delay_path8, filt_path1_s, filt_path2_s, filt_path4_s, filt_path8_s):
    if not os.path.exists(dest_path):
        print('Creating d2 folder')
        os.mkdir(dest_path)
    if not os.path.exists(filt_path1):
        print('Creating rt 1 folder (double)')
        os.mkdir(filt_path1)
    if not os.path.exists(filt_path2):
        print('Creating rt 2 folder (double)')
        os.mkdir(filt_path2)
    if not os.path.exists(filt_path4):
        print('Creating rt 4 folder (double)')
        os.mkdir(filt_path4)
    if not os.path.exists(filt_path8):
        print('Creating rt 8 folder (double)')
        os.mkdir(filt_path8)
    if not os.path.exists(delay_path1):
        print('Creating rt 1 delay folder')
        os.mkdir(delay_path1)
    if not os.path.exists(delay_path2):
        print('Creating rt 2 delay folder')
        os.mkdir(delay_path2)
    if not os.path.exists(delay_path4):
        print('Creating rt 4 delay folder')
        os.mkdir(delay_path4)
    if not os.path.exists(delay_path8):
        print('Creating rt 8 delay folder')
        os.mkdir(delay_path8)
    if not os.path.exists(filt_path1_s):
        print('Creating rt 1 folder (single)')
        os.mkdir(filt_path1_s)
    if not os.path.exists(filt_path2_s):
        print('Creating rt 2 folder (single)')
        os.mkdir(filt_path2_s)
    if not os.path.exists(filt_path4_s):
        print('Creating rt 4 folder (single)')
        os.mkdir(filt_path4_s)
    if not os.path.exists(filt_path8_s):
        print('Creating rt 8 folder (single)')
        os.mkdir(filt_path8_s)
    if not os.path.exists(Path(dest_path / 'hist_data_double')):
        print('Creating histogram data folder')
        os.mkdir(Path(dest_path / 'hist_data_double'))
    if not os.path.exists(Path(dest_path / 'plots')):
        print('Creating plots folder')
        os.mkdir(Path(dest_path / 'plots'))
    if not os.path.exists(Path(dest_path / 'calculations_double')):
        print('Creating calculations folder')
        os.mkdir(Path(dest_path / 'calculations_double'))
    if not os.path.exists(Path(dest_path / 'unusable_data')):
        print('Creating unusable data folder')
        os.mkdir(Path(dest_path / 'unusable_data'))


# Makes arrays of existing single and double spe files
def initial_arrays(single_path, filt_path1_s, delay_path1):
    single_file_array = np.array([])
    single_file_array2 = np.array([])
    double_file_array = np.array([])

    print('Checking single spe files...')
    for i in range(99999):                                  # Makes array of all spe file names
        file_name = 'D2--waveforms--%05d.txt' % i
        if os.path.isfile(single_path / file_name):
            single_file_array = np.append(single_file_array, i)

    print('Checking existing single spe files...')
    for i in range(99999):                                  # Makes array of all spe file names
        file_name = 'D2--waveforms--%05d.txt' % i
        if os.path.isfile(filt_path1_s / file_name):
            single_file_array2 = np.append(single_file_array2, i)

    print('Checking existing double spe files...')
    for filename in os.listdir(delay_path1):                # Checks for existing double spe files
        print(filename, 'is a file')
        files_added = filename[15:27]
        double_file_array = np.append(double_file_array, files_added)

    return single_file_array, single_file_array2, double_file_array


# Adds two random spe files with a given delay
def add_spe(single_file_array, double_file_array, delay, delay_path1, nloops, single_path, nhdr):
    if len(double_file_array) < nloops:
        file_1 = single_file_array[np.random.randint(len(single_file_array))]
        file_2 = single_file_array[np.random.randint(len(single_file_array))]
        file_name_1 = str(single_path / 'D2--waveforms--%05d.txt') % file_1
        file_name_2 = str(single_path / 'D2--waveforms--%05d.txt') % file_2
        files_added = '%05d--%05d' % (file_1, file_2)

        t1, v1, hdr1 = rw(file_name_1, nhdr)
        t2, v2, hdr2 = rw(file_name_2, nhdr)
        for j in range(len(t1)):
            t1[j] = float(format(t1[j], '.4e'))
        for j in range(len(t2)):
            t2[j] = float(format(t2[j], '.4e'))

        time_int = float(format(t1[1] - t1[0], '.4e'))
        delay_amt = int(delay / time_int) * time_int

        try:
            avg = calculate_average(t2, v2)
            time_1, time_2 = calculate_t1_t2(t2, v2)

            for i in range(np.argmin(np.abs(t2 - time_1))[0]):
                v2[i] = avg

            for i in range(np.argmin(np.abs(t2 - time_2))[0], len(v2) - 1):
                v2[i] = avg

            if min(t1) < min(t2):
                t1 += delay_amt
            else:
                t2 += delay_amt

            if min(t1) < min(t2):
                idx1 = np.where(t1 == min(t2))[0][0]
                for j in range(idx1):
                    t1 = np.append(t1, float(format(max(t1) + time_int, '.4e')))
                    t2 = np.insert(t2, 0, float(format(min(t2) + time_int, '.4e')))
                    v1 = np.append(v1, 0)
                    v2 = np.insert(v2, 0, 0)
            elif min(t1) > min(t2):
                idx2 = np.where(t2 == min(t1))[0][0]
                for j in range(idx2):
                    t1 = np.insert(t1, 0, float(format(min(t1) - time_int, '.4e')))
                    t2 = np.append(t2, float(format(max(t2) + time_int, '.4e')))
                    v1 = np.insert(v1, 0, 0)
                    v2 = np.append(v2, 0)
            else:
                pass

            t = t1
            v = np.add(v1, v2)
            file_name = 'D2--waveforms--%s.txt' % files_added
            ww(t, v, delay_path1 / file_name, hdr1)
            double_file_array = np.append(double_file_array, files_added)
            print('Added files #%05d & #%05d' % (file_1, file_2))

        except Exception:
            pass

    return double_file_array


# Creates set of single spe files to compare to doubles
def single_set(single_file_array, single_file_array2, nloops, single_path, filt_path1_s):
    if len(single_file_array2) < nloops:
        file = single_file_array[np.random.randint(len(single_file_array))]
        file_num = '%05d' % file

        if not os.path.isfile(filt_path1_s / str('D2--waveforms--%s.txt') % file_num):
            single_file_array2 = np.append(single_file_array2, file_num)
            t, v, hdr = rw(str(single_path / 'D2--waveforms--%s.txt') % file_num, nhdr)
            ww(t, v, str(filt_path1_s / 'D2--waveforms--%s.txt') % file_num, hdr)
            print('File #%05d added' % file)

    return single_file_array2


# Calculates and saves waveforms with 1x, 2x, 4x, and 8x the rise time
def shaping(save_name1, save_name2, save_name4, save_name8, item, fsps, nhdr):
    tau_2 = 1.271e-08
    tau_4 = 1.0479999999999999e-08
    tau_8 = 2.7539999999999997e-08
    factor2 = 2.6769456128607705
    factor4 = 3.720902601689933
    factor8 = 6.301083740858239

    if os.path.isfile(save_name2):
        print('File #%05d in rt_2 folder' % item)
    else:
        if os.path.isfile(save_name1):
            t1, v1, hdr = rw(save_name1, nhdr)
            v2 = lowpass_filter(v1, tau_2, fsps)
            v2_gain = v2 * factor2
            t2 = t1
            ww(t2, v2_gain, save_name2, hdr)
            print('File #%05d in rt_2 folder' % item)
        else:
            pass

    if os.path.isfile(save_name4):
        print('File #%05d in rt_4 folder' % item)
    else:
        if os.path.isfile(save_name2):
            t2, v2, hdr = rw(save_name2, nhdr)
            v4 = lowpass_filter(v2, tau_4, fsps)
            v4_gain = v4 * factor4
            t4 = t2
            ww(t4, v4_gain, save_name4, hdr)
            print('File #%05d in rt_4 folder' % item)
        else:
            pass

    if os.path.isfile(save_name8):
        print('File #%05d in rt_8 folder' % item)
    else:
        if os.path.isfile(save_name4):
            t4, v4, hdr = rw(save_name4, nhdr)
            v8 = lowpass_filter(v4, tau_8, fsps)
            v8_gain = v8 * factor8
            t8 = t4
            ww(t8, v8_gain, save_name8, hdr)
            print('File #%05d in rt_8 folder' % item)
        else:
            pass


def delay_names(delay_folder):
    if delay_folder == 'no_delay':
        delay_name = 'no delay'
    elif delay_folder == '0.5x_rt':
        delay_name = '1.52 ns delay'
    elif delay_folder == '1x_rt':
        delay_name = '3.04 ns delay'
    elif delay_folder == '1.5x_rt':
        delay_name = '4.56 ns delay'
    elif delay_folder == '2x_rt':
        delay_name = '6.08 ns delay'
    elif delay_folder == '2.5x_rt':
        delay_name = '7.6 ns delay'
    elif delay_folder == '3x_rt':
        delay_name = '9.12 ns delay'
    elif delay_folder == '3.5x_rt':
        delay_name = '10.6 ns delay'
    elif delay_folder == '4x_rt':
        delay_name = '12.2 ns delay'
    elif delay_folder == '4.5x_rt':
        delay_name = '13.7 ns delay'
    elif delay_folder == '5x_rt':
        delay_name = '15.2 ns delay'
    elif delay_folder == '5.5x_rt':
        delay_name = '16.7 ns delay'
    elif delay_folder == '6x_rt':
        delay_name = '18.2 ns delay'
    elif delay_folder == '40_ns':
        delay_name = '40 ns delay'
    elif delay_folder == '80_ns':
        delay_name = '80 ns delay'
    else:
        delay_name = ''

    return delay_name
