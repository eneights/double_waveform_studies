from p3_functions import *


# Downsamples and digitizes double spe waveforms, then calculates charge, amplitude, and FWHM
def double_spe_studies(date, filter_band, nhdr, delay_folder, fsps, fsps_new, noise, r):
    gen_path, save_path, data_path, dest_path, filt_path1, filt_path2, filt_path4, filt_path8 = \
        initialize_folders(date, filter_band)
    make_folders(dest_path, filt_path1, filt_path2, filt_path4, filt_path8, fsps_new, delay_folder)

    single_file_array, double_file_array = initial_arrays(Path(Path(dest_path / 'rt_1_single_2') / 'raw'),
                                                          Path(filt_path1 / 'raw' / delay_folder))

    copy_s_waveforms(single_file_array, data_path, dest_path, nhdr)
    copy_d_waveforms(double_file_array, data_path, filt_path1, filt_path2, filt_path4, filt_path8, delay_folder, nhdr)

    down_dig(single_file_array, double_file_array, filt_path1, filt_path2, filt_path4, filt_path8, dest_path,
             delay_folder, fsps, fsps_new, noise, nhdr)

    # Creates double spe arrays for charge, amplitude, and FWHM
    charge_array, amplitude_array, fwhm_array = make_arrays(double_file_array, 'rt_1', delay_folder, dest_path,
                                                            nhdr, r, fsps_new)
    charge_array_2, amplitude_array_2, fwhm_array_2 = make_arrays(double_file_array, 'rt_2', delay_folder,
                                                                  dest_path, nhdr, r, fsps_new)
    charge_array_4, amplitude_array_4, fwhm_array_4 = make_arrays(double_file_array, 'rt_4', delay_folder,
                                                                  dest_path, nhdr, r, fsps_new)
    charge_array_8, amplitude_array_8, fwhm_array_8 = make_arrays(double_file_array, 'rt_8', delay_folder,
                                                                  dest_path, nhdr, r, fsps_new)

    # Creates single spe arrays for charge, amplitude, and FWHM
    charge_array_s, amplitude_array_s, fwhm_array_s = make_arrays_s(single_file_array, dest_path, 'rt_1', nhdr, r,
                                                                    fsps_new)
    charge_array_2_s, amplitude_array_2_s, fwhm_array_2_s = make_arrays_s(single_file_array, dest_path, 'rt_2', nhdr, r,
                                                                          fsps_new)
    charge_array_4_s, amplitude_array_4_s, fwhm_array_4_s = make_arrays_s(single_file_array, dest_path, 'rt_4', nhdr, r,
                                                                          fsps_new)
    charge_array_8_s, amplitude_array_8_s, fwhm_array_8_s = make_arrays_s(single_file_array, dest_path, 'rt_8', nhdr, r,
                                                                          fsps_new)

    # Creates double spe histograms for charge, amplitude, and FWHM
    print('Creating double spe histograms...')
    plot_histogram(charge_array, dest_path, 75, 'Charge', 'Charge', 's*bit/ohm', 'charge_double_rt1_' +
                   str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(charge_array_2, dest_path, 75, 'Charge', 'Charge', 's*bit/ohm', 'charge_double_rt2_' +
                   str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(charge_array_4, dest_path, 75, 'Charge', 'Charge', 's*bit/ohm', 'charge_double_rt4_' +
                   str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(charge_array_8, dest_path, 75, 'Charge', 'Charge', 's*bit/ohm', 'charge_double_rt8_' +
                   str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(amplitude_array, dest_path, 75, 'Amplitude', 'Amplitude', 'bits', 'amp_double_rt1_' +
                   str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(amplitude_array_2, dest_path, 75, 'Amplitude', 'Amplitude', 'bits', 'amp_double_rt2_' +
                   str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(amplitude_array_4, dest_path, 75, 'Amplitude', 'Amplitude', 'bits', 'amp_double_rt4_' +
                   str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(amplitude_array_8, dest_path, 75, 'Amplitude', 'Amplitude', 'bits', 'amp_double_rt8_' +
                   str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(fwhm_array, dest_path, 75, 'Time', 'FWHM', 's', 'fwhm_double_rt1_' + str(delay_folder) + '_' +
                   str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(fwhm_array_2, dest_path, 75, 'Time', 'FWHM', 's', 'fwhm_double_rt2_' + str(delay_folder) + '_' +
                   str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(fwhm_array_4, dest_path, 75, 'Time', 'FWHM', 's', 'fwhm_double_rt4_' + str(delay_folder) + '_' +
                   str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(fwhm_array_8, dest_path, 75, 'Time', 'FWHM', 's', 'fwhm_double_rt8_' + str(delay_folder) + '_' +
                   str(int(fsps_new / 1e6)) + '_Msps', fsps_new)

    # Creates single spe histograms for charge, amplitude, and FWHM
    print('Creating single spe histograms...')
    plot_histogram(charge_array_s, dest_path, 75, 'Charge', 'Charge', 's*bit/ohm', 'charge_single_rt1' + '_' +
                   str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(charge_array_2_s, dest_path, 75, 'Charge', 'Charge', 's*bit/ohm', 'charge_single_rt2' + '_' +
                   str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(charge_array_4_s, dest_path, 75, 'Charge', 'Charge', 's*bit/ohm', 'charge_single_rt4' + '_' +
                   str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(charge_array_8_s, dest_path, 75, 'Charge', 'Charge', 's*bit/ohm', 'charge_single_rt8' + '_' +
                   str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(amplitude_array_s, dest_path, 75, 'Amplitude', 'Amplitude', 'bits', 'amp_single_rt1' + '_' +
                   str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(amplitude_array_2_s, dest_path, 75, 'Amplitude', 'Amplitude', 'bits', 'amp_single_rt2' + '_' +
                   str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(amplitude_array_4_s, dest_path, 75, 'Amplitude', 'Amplitude', 'bits', 'amp_single_rt4' + '_' +
                   str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(amplitude_array_8_s, dest_path, 75, 'Amplitude', 'Amplitude', 'bits', 'amp_single_rt8' + '_' +
                   str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(fwhm_array_s, dest_path, 75, 'Time', 'FWHM', 's', 'fwhm_single_rt1' + '_' +
                   str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(fwhm_array_2_s, dest_path, 75, 'Time', 'FWHM', 's', 'fwhm_single_rt2' + '_' +
                   str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(fwhm_array_4_s, dest_path, 75, 'Time', 'FWHM', 's', 'fwhm_single_rt4' + '_' +
                   str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(fwhm_array_8_s, dest_path, 75, 'Time', 'FWHM', 's', 'fwhm_single_rt8' + '_' +
                   str(int(fsps_new / 1e6)) + '_Msps', fsps_new)

    # Creates double & single spe histograms on same plot
    print('Creating histograms...')
    plot_double_hist(dest_path, 75, 'Charge', 'Charge', 's*bit/ohm', 'charge_single_rt1', 'charge_double_rt1_' +
                     str(delay_folder), 'charge_rt1_' + str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps',
                     fsps_new)
    plot_double_hist(dest_path, 75, 'Charge', 'Charge', 's*bit/ohm', 'charge_single_rt2', 'charge_double_rt2_' +
                     str(delay_folder), 'charge_rt2_' + str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps',
                     fsps_new)
    plot_double_hist(dest_path, 75, 'Charge', 'Charge', 's*bit/ohm', 'charge_single_rt4', 'charge_double_rt4_' +
                     str(delay_folder), 'charge_rt4_' + str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps',
                     fsps_new)
    plot_double_hist(dest_path, 75, 'Charge', 'Charge', 's*bit/ohm', 'charge_single_rt8', 'charge_double_rt8_' +
                     str(delay_folder), 'charge_rt8_' + str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps',
                     fsps_new)
    plot_double_hist(dest_path, 75, 'Amplitude', 'Amplitude', 'bits', 'amp_single_rt1', 'amp_double_rt1_' +
                     str(delay_folder), 'amp_rt1_' + str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps',
                     fsps_new)
    plot_double_hist(dest_path, 75, 'Amplitude', 'Amplitude', 'bits', 'amp_single_rt2', 'amp_double_rt2_'
                     + str(delay_folder), 'amp_rt2_' + str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps',
                     fsps_new)
    plot_double_hist(dest_path, 75, 'Amplitude', 'Amplitude', 'bits', 'amp_single_rt4', 'amp_double_rt4_'
                     + str(delay_folder), 'amp_rt4_' + str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps',
                     fsps_new)
    plot_double_hist(dest_path, 75, 'Amplitude', 'Amplitude', 'bits', 'amp_single_rt8', 'amp_double_rt8_'
                     + str(delay_folder), 'amp_rt8_' + str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps',
                     fsps_new)
    plot_double_hist(dest_path, 75, 'Time', 'FWHM', 's', 'fwhm_single_rt1', 'fwhm_double_rt1_' + str(delay_folder),
                     'fwhm_rt1_' + str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_double_hist(dest_path, 75, 'Time', 'FWHM', 's', 'fwhm_single_rt2', 'fwhm_double_rt2_' + str(delay_folder),
                     'fwhm_rt2_' + str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_double_hist(dest_path, 75, 'Time', 'FWHM', 's', 'fwhm_single_rt4', 'fwhm_double_rt4_' + str(delay_folder),
                     'fwhm_rt4_' + str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_double_hist(dest_path, 75, 'Time', 'FWHM', 's', 'fwhm_single_rt8', 'fwhm_double_rt8_' + str(delay_folder),
                     'fwhm_rt8_' + str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps', fsps_new)




    filename1 = 'fwhm_single_' + shaping
    filename2 = 'fwhm_double_' + shaping + '_no_delay'
    filename3 = 'fwhm_double_' + shaping + '_0.5x_rt'
    filename4 = 'fwhm_double_' + shaping + '_1x_rt'
    filename5 = 'fwhm_double_' + shaping + '_1.5x_rt'
    filename6 = 'fwhm_double_' + shaping + '_2x_rt'
    filename7 = 'fwhm_double_' + shaping + '_2.5x_rt'
    filename8 = 'fwhm_double_' + shaping + '_3x_rt'
    filename9 = 'fwhm_double_' + shaping + '_3.5x_rt'
    filename10 = 'fwhm_double_' + shaping + '_4x_rt'
    filename11 = 'fwhm_double_' + shaping + '_4.5x_rt'
    filename12 = 'fwhm_double_' + shaping + '_5x_rt'
    filename13 = 'fwhm_double_' + shaping + '_5.5x_rt'
    filename14 = 'fwhm_double_' + shaping + '_6x_rt'
    filename15 = 'fwhm_double_' + shaping + '_40_ns'
    filename16 = 'fwhm_double_' + shaping + '_80_ns'
    filename17 = 'amp_single_' + shaping
    filename18 = 'amp_double_' + shaping + '_no_delay'
    filename19 = 'charge_single_' + shaping
    filename20 = 'charge_double_' + shaping + '_no_delay'
    filename21 = 'charge_double_' + shaping + '_0.5x_rt'
    filename22 = 'charge_double_' + shaping + '_1x_rt'
    filename23 = 'charge_double_' + shaping + '_1.5x_rt'
    filename24 = 'charge_double_' + shaping + '_2x_rt'
    filename25 = 'charge_double_' + shaping + '_2.5x_rt'
    filename26 = 'charge_double_' + shaping + '_3x_rt'
    filename27 = 'charge_double_' + shaping + '_3.5x_rt'
    filename28 = 'charge_double_' + shaping + '_4x_rt'
    filename29 = 'charge_double_' + shaping + '_4.5x_rt'
    filename30 = 'charge_double_' + shaping + '_5x_rt'
    filename31 = 'charge_double_' + shaping + '_5.5x_rt'
    filename32 = 'charge_double_' + shaping + '_6x_rt'
    filename33 = 'charge_double_' + shaping + '_40_ns'
    filename34 = 'charge_double_' + shaping + '_80_ns'

    array_single_fwhm = read_hist_file(dest_path / 'hist_data', filename1, fsps_new)
    array_double_fwhm_nd = read_hist_file(dest_path / 'hist_data', filename2, fsps_new)
    array_double_fwhm__5x = read_hist_file(dest_path / 'hist_data', filename3, fsps_new)
    array_double_fwhm_1x = read_hist_file(dest_path / 'hist_data', filename4, fsps_new)
    array_double_fwhm_15x = read_hist_file(dest_path / 'hist_data', filename5, fsps_new)
    array_double_fwhm_2x = read_hist_file(dest_path / 'hist_data', filename6, fsps_new)
    array_double_fwhm_25x = read_hist_file(dest_path / 'hist_data', filename7, fsps_new)
    array_double_fwhm_3x = read_hist_file(dest_path / 'hist_data', filename8, fsps_new)
    array_double_fwhm_35x = read_hist_file(dest_path / 'hist_data', filename9, fsps_new)
    array_double_fwhm_4x = read_hist_file(dest_path / 'hist_data', filename10, fsps_new)
    array_double_fwhm_45x = read_hist_file(dest_path / 'hist_data', filename11, fsps_new)
    array_double_fwhm_5x = read_hist_file(dest_path / 'hist_data', filename12, fsps_new)
    array_double_fwhm_55x = read_hist_file(dest_path / 'hist_data', filename13, fsps_new)
    array_double_fwhm_6x = read_hist_file(dest_path / 'hist_data', filename14, fsps_new)
    array_double_fwhm_40 = read_hist_file(dest_path / 'hist_data', filename15, fsps_new)
    array_double_fwhm_80 = read_hist_file(dest_path / 'hist_data', filename16, fsps_new)
    array_single_amp = read_hist_file(dest_path / 'hist_data', filename17, fsps_new)
    array_double_amp_nd = read_hist_file(dest_path / 'hist_data', filename18, fsps_new)
    array_single_charge = read_hist_file(dest_path / 'hist_data', filename19, fsps_new)
    array_double_charge_nd = read_hist_file(dest_path / 'hist_data', filename20, fsps_new)
    array_double_charge__5x = read_hist_file(dest_path / 'hist_data', filename21, fsps_new)
    array_double_charge_1x = read_hist_file(dest_path / 'hist_data', filename22, fsps_new)
    array_double_charge_15x = read_hist_file(dest_path / 'hist_data', filename23, fsps_new)
    array_double_charge_2x = read_hist_file(dest_path / 'hist_data', filename24, fsps_new)
    array_double_charge_25x = read_hist_file(dest_path / 'hist_data', filename25, fsps_new)
    array_double_charge_3x = read_hist_file(dest_path / 'hist_data', filename26, fsps_new)
    array_double_charge_35x = read_hist_file(dest_path / 'hist_data', filename27, fsps_new)
    array_double_charge_4x = read_hist_file(dest_path / 'hist_data', filename28, fsps_new)
    array_double_charge_45x = read_hist_file(dest_path / 'hist_data', filename29, fsps_new)
    array_double_charge_5x = read_hist_file(dest_path / 'hist_data', filename30, fsps_new)
    array_double_charge_55x = read_hist_file(dest_path / 'hist_data', filename31, fsps_new)
    array_double_charge_6x = read_hist_file(dest_path / 'hist_data', filename32, fsps_new)
    array_double_charge_40 = read_hist_file(dest_path / 'hist_data', filename33, fsps_new)
    array_double_charge_80 = read_hist_file(dest_path / 'hist_data', filename34, fsps_new)

    mean_single_fwhm = np.mean(array_single_fwhm)
    mean_double_fwhm_nd = np.mean(array_double_fwhm_nd)
    mean_double_fwhm__5x = np.mean(array_double_fwhm__5x)
    mean_double_fwhm_1x = np.mean(array_double_fwhm_1x)
    mean_double_fwhm_15x = np.mean(array_double_fwhm_15x)
    mean_double_fwhm_2x = np.mean(array_double_fwhm_2x)
    mean_double_fwhm_25x = np.mean(array_double_fwhm_25x)
    mean_double_fwhm_3x = np.mean(array_double_fwhm_3x)
    mean_double_fwhm_35x = np.mean(array_double_fwhm_35x)
    mean_double_fwhm_4x = np.mean(array_double_fwhm_4x)
    mean_double_fwhm_45x = np.mean(array_double_fwhm_45x)
    mean_double_fwhm_5x = np.mean(array_double_fwhm_5x)
    mean_double_fwhm_55x = np.mean(array_double_fwhm_55x)
    mean_double_fwhm_6x = np.mean(array_double_fwhm_6x)
    mean_double_fwhm_40 = np.mean(array_double_fwhm_40)
    mean_double_fwhm_80 = np.mean(array_double_fwhm_80)
    mean_single_amp = np.mean(array_single_amp)
    mean_double_amp_nd = np.mean(array_double_amp_nd)
    mean_single_charge = np.mean(array_single_charge)
    mean_double_charge_nd = np.mean(array_double_charge_nd)
    mean_double_charge__5x = np.mean(array_double_charge_5x)
    mean_double_charge_1x = np.mean(array_double_charge_1x)
    mean_double_charge_15x = np.mean(array_double_charge_15x)
    mean_double_charge_2x = np.mean(array_double_charge_2x)
    mean_double_charge_25x = np.mean(array_double_charge_25x)
    mean_double_charge_3x = np.mean(array_double_charge_3x)
    mean_double_charge_35x = np.mean(array_double_charge_35x)
    mean_double_charge_4x = np.mean(array_double_charge_4x)
    mean_double_charge_45x = np.mean(array_double_charge_45x)
    mean_double_charge_5x = np.mean(array_double_charge_5x)
    mean_double_charge_55x = np.mean(array_double_charge_55x)
    mean_double_charge_6x = np.mean(array_double_charge_6x)
    mean_double_charge_40 = np.mean(array_double_charge_40)
    mean_double_charge_80 = np.mean(array_double_charge_80)

    std_single_fwhm = np.std(array_single_fwhm)
    std_double_fwhm_nd = np.std(array_double_fwhm_nd)
    std_double_fwhm__5x = np.std(array_double_fwhm__5x)
    std_double_fwhm_1x = np.std(array_double_fwhm_1x)
    std_double_fwhm_15x = np.std(array_double_fwhm_15x)
    std_double_fwhm_2x = np.std(array_double_fwhm_2x)
    std_double_fwhm_25x = np.std(array_double_fwhm_25x)
    std_double_fwhm_3x = np.std(array_double_fwhm_3x)
    std_double_fwhm_35x = np.std(array_double_fwhm_35x)
    std_double_fwhm_4x = np.std(array_double_fwhm_4x)
    std_double_fwhm_45x = np.std(array_double_fwhm_45x)
    std_double_fwhm_5x = np.std(array_double_fwhm_5x)
    std_double_fwhm_55x = np.std(array_double_fwhm_55x)
    std_double_fwhm_6x = np.std(array_double_fwhm_6x)
    std_double_fwhm_40 = np.mean(array_double_fwhm_40)
    std_double_fwhm_80 = np.mean(array_double_fwhm_80)
    std_single_amp = np.std(array_single_amp)
    std_double_amp_nd = np.std(array_double_amp_nd)
    std_single_charge = np.std(array_single_charge)
    std_double_charge_nd = np.std(array_double_charge_nd)
    std_double_charge__5x = np.std(array_double_charge__5x)
    std_double_charge_1x = np.std(array_double_charge_1x)
    std_double_charge_15x = np.std(array_double_charge_15x)
    std_double_charge_2x = np.std(array_double_charge_2x)
    std_double_charge_25x = np.std(array_double_charge_25x)
    std_double_charge_3x = np.std(array_double_charge_3x)
    std_double_charge_35x = np.std(array_double_charge_35x)
    std_double_charge_4x = np.std(array_double_charge_4x)
    std_double_charge_45x = np.std(array_double_charge_45x)
    std_double_charge_5x = np.std(array_double_charge_5x)
    std_double_charge_55x = np.std(array_double_charge_55x)
    std_double_charge_6x = np.std(array_double_charge_6x)
    std_double_charge_40 = np.mean(array_double_charge_40)
    std_double_charge_80 = np.mean(array_double_charge_80)

    print('Making plots...')

    if shaping == 'rt1':
        start1 = 5
        end1 = 35
        factor1 = 10 ** -9
        start2 = 10
        end2 = 125
        factor2 = 10 ** -9
        start3 = 50
        end3 = 550
        factor3 = 1
    elif shaping == 'rt2':
        start1 = 10
        end1 = 60
        factor1 = 10 ** -9
        start2 = 10
        end2 = 125
        factor2 = 10 ** -9
        start3 = 25
        end3 = 200
        factor3 = 1
    else:
        start1 = 20
        end1 = 80
        factor1 = 10 ** -9
        start2 = 10
        end2 = 125
        factor2 = 10 ** -9
        start3 = 20
        end3 = 150
        factor3 = 1

    false_spes_vs_delay(start1, end1, factor1, 'fwhm', 'FWHM', 's', fsps_new, mean_single_fwhm, mean_double_fwhm__5x,
                        mean_double_fwhm_1x, mean_double_fwhm_15x, mean_double_fwhm_2x, mean_double_fwhm_25x,
                        mean_double_fwhm_3x, mean_double_fwhm_35x, mean_double_fwhm_4x, mean_double_fwhm_45x,
                        mean_double_fwhm_5x, mean_double_fwhm_55x, mean_double_fwhm_6x, mean_double_fwhm_40,
                        mean_double_fwhm_80, std_single_fwhm, std_double_fwhm__5x, std_double_fwhm_1x,
                        std_double_fwhm_15x, std_double_fwhm_2x, std_double_fwhm_25x, std_double_fwhm_3x,
                        std_double_fwhm_35x, std_double_fwhm_4x, std_double_fwhm_45x, std_double_fwhm_5x,
                        std_double_fwhm_55x, std_double_fwhm_6x, std_double_fwhm_40, std_double_fwhm_80, dest_path,
                        shaping)

    false_spes_mpes(start2, end2, factor2, 'charge', 'Charge', 's*bit/ohm', mean_single_charge, mean_double_charge_nd,
                    std_single_charge, std_double_charge_nd, fsps_new, dest_path, shaping)
    false_spes_mpes(start3, end3, factor3, 'amp', 'Amplitude', 'bits', mean_single_amp, mean_double_amp_nd,
                    std_single_amp, std_double_amp_nd, fsps_new, dest_path, shaping)

    roc_graphs(start1, end1, factor1, fsps_new, shaping, 'fwhm', 'FWHM', mean_single_fwhm, mean_double_fwhm_nd,
               mean_double_fwhm__5x, mean_double_fwhm_1x, mean_double_fwhm_15x, mean_double_fwhm_2x,
               mean_double_fwhm_25x, mean_double_fwhm_3x, mean_double_fwhm_35x, mean_double_fwhm_4x,
               mean_double_fwhm_45x, mean_double_fwhm_5x, mean_double_fwhm_55x, mean_double_fwhm_6x,
               mean_double_fwhm_40, mean_double_fwhm_80, std_single_fwhm, std_double_fwhm_nd, std_double_fwhm__5x,
               std_double_fwhm_1x, std_double_fwhm_15x, std_double_fwhm_2x, std_double_fwhm_25x, std_double_fwhm_3x,
               std_double_fwhm_35x, std_double_fwhm_4x, std_double_fwhm_45x, std_double_fwhm_5x, std_double_fwhm_55x,
               std_double_fwhm_6x, std_double_fwhm_40, std_double_fwhm_80, dest_path)
    roc_graphs(start2, end2, factor2, fsps_new, shaping, 'charge', 'Charge', mean_single_charge, mean_double_charge_nd,
               mean_double_charge__5x, mean_double_charge_1x, mean_double_charge_15x, mean_double_charge_2x,
               mean_double_charge_25x, mean_double_charge_3x, mean_double_charge_35x, mean_double_charge_4x,
               mean_double_charge_45x, mean_double_charge_5x, mean_double_charge_55x, mean_double_charge_6x,
               mean_double_charge_40, mean_double_charge_80, std_single_charge, std_double_charge_nd,
               std_double_charge__5x, std_double_charge_1x, std_double_charge_15x, std_double_charge_2x,
               std_double_charge_25x, std_double_charge_3x, std_double_charge_35x, std_double_charge_4x,
               std_double_charge_45x, std_double_charge_5x, std_double_charge_55x, std_double_charge_6x,
               std_double_charge_40, std_double_charge_80, dest_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog="double_spe_studies", description="Creating double spe D3")
    parser.add_argument("--date", type=int, help='date of data acquisition (default=20190513)', default=20190513)
    parser.add_argument("--fil_band", type=str, help='folder name for data (default=full_bdw_no_nf)',
                        default='full_bdw_no_nf')
    parser.add_argument("--nhdr", type=int, help='number of header lines to skip in raw file (default=5)', default=5)
    parser.add_argument("--delay_folder", type=str, help='folder name for delay (default=no_delay)', default='no_delay')
    parser.add_argument("--fsps", type=float, help='samples per second (Hz) (default=20000000000.)',
                        default=20000000000.)
    parser.add_argument("--fsps_new", type=float, help='new samples per second (Hz) (default=500000000.)',
                        default=500000000.)
    parser.add_argument("--noise", type=float, help='noise to add (bits) (default=3.30)', default=3.30)
    parser.add_argument("--r", type=int, help='resistance in ohms (default=50)', default=50)
    args = parser.parse_args()

    double_spe_studies(args.date, args.fil_band, args.nhdr, args.delay_folder, args.fsps, args.fsps_new, args.noise,
                       args.r)