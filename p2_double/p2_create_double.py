from p2_functions import *


# Creates data set of double spe waveforms (and set of single spe waveforms for comparison)
def create_double_spe(nloops, date, filter_band, nhdr, delay, delay_folder, fsps):
    gen_path, save_path, dest_path, single_path, filt_path1, filt_path2, filt_path4, filt_path8, delay_path1, \
    delay_path2, delay_path4, delay_path8, filt_path1_s, filt_path2_s, filt_path4_s, filt_path8_s = \
        initialize_folders(date, filter_band, delay_folder)
    make_folders(dest_path, filt_path1, filt_path2, filt_path4, filt_path8, delay_path1, delay_path2, delay_path4,
                 delay_path8, filt_path1_s, filt_path2_s, filt_path4_s, filt_path8_s)

    single_file_array, single_file_array2, double_file_array = initial_arrays(single_path, filt_path1_s, delay_path1)

    # Creates double spe files
    print('Adding double files...')
    for i in range(nloops):
        double_file_array = add_spe(single_file_array, double_file_array, delay, delay_path1, nloops, single_path, nhdr)

    # Creates single spe files
    print('Adding single files...')
    for i in range(nloops):
        single_file_array2 = single_set(single_file_array, single_file_array2, nloops, single_path, filt_path1_s, nhdr)

    # Shapes single and double waveforms
    for item in single_file_array:
        save_name1 = str(filt_path1_s / 'D2--waveforms--%s.txt') % item
        save_name2 = str(filt_path2_s / 'D2--waveforms--%s.txt') % item
        save_name4 = str(filt_path4_s / 'D2--waveforms--%s.txt') % item
        save_name8 = str(filt_path8_s / 'D2--waveforms--%s.txt') % item

        shaping(save_name1, save_name2, save_name4, save_name8, item, fsps, nhdr)

    for item in double_file_array:
        save_name1 = str(delay_path1 / 'D2--waveforms--%s.txt') % item
        save_name2 = str(delay_path2 / 'D2--waveforms--%s.txt') % item
        save_name4 = str(delay_path4 / 'D2--waveforms--%s.txt') % item
        save_name8 = str(delay_path8 / 'D2--waveforms--%s.txt') % item

        shaping(save_name1, save_name2, save_name4, save_name8, item, fsps, nhdr)

    # Creates name of delay folder
    delay_name = delay_names(delay_folder)

    # Plots average waveform for 1x rise time
    print('Calculating rt_1 average waveform...')
    average_waveform(double_file_array, dest_path, 'rt_1', 'No Shaping', delay_path1, delay_name, delay_folder, nhdr)

    # Plots average waveform for 2x rise time
    print('Calculating rt_2 average waveform...')
    average_waveform(double_file_array, dest_path, 'rt_2', '2x Shaping', delay_path2, delay_name, delay_folder, nhdr)

    # Plots average waveform for 4x rise time
    print('Calculating rt_4 average waveform...')
    average_waveform(double_file_array, dest_path, 'rt_4', '4x Shaping', delay_path4, delay_name, delay_folder, nhdr)

    # Plots average waveform for 8x rise time
    print('Calculating rt_8 average waveform...')
    average_waveform(double_file_array, dest_path, 'rt_8', '8x Shaping', delay_path8, delay_name, delay_folder, nhdr)

    # Calculates 10-90 rise times for each double spe waveform and puts them into arrays
    print('Doing calculations...')
    rt_1_array, rt_2_array, rt_4_array, rt_8_array = make_arrays(double_file_array, delay_path1, delay_path2,
                                                                 delay_path4, delay_path8, dest_path, nhdr)

    # Creates histograms of 10-90 rise times for 1x, 2x, 4x, and 8x the initial rise time for double spe waveforms
    p2_hist(rt_1_array, rt_2_array, rt_4_array, rt_8_array, dest_path, 100, delay_name, delay_folder)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog="create_double_spe", description="Adds spe waveforms")
    parser.add_argument("--nloops", type=int, help='number of double spe files to create (default=1000)', default=1000)
    parser.add_argument("--date", type=int, help='date of data acquisition (default=20190513)', default=20190513)
    parser.add_argument("--fil_band", type=str, help='folder name for data (default=full_bdw_no_nf)',
                        default='full_bdw_no_nf')
    parser.add_argument("--nhdr", type=int, help='number of header lines to skip in raw file (default=5)', default=5)
    parser.add_argument("--delay", type=float, help='delay time (s) (default=0.)', default=0.)
    parser.add_argument("--delay_folder", type=str, help='folder name for delay (default=no_delay)', default='no_delay')
    parser.add_argument("--fsps", type=float, help='samples per second (Hz) (default=20000000000.)',
                        default=20000000000.)
    args = parser.parse_args()

    create_double_spe(args.nloops, args.date, args.fil_band, args.nhdr, args.delay, args.delay_folder, args.fsps)