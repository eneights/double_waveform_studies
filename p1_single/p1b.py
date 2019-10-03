from p1_functions import *


def p1b(start, end, dest_path, nhdr):

    charge_array, amplitude_array, fwhm_array, rise1090_array, rise2080_array, fall1090_array, fall2080_array, \
        time10_array, time20_array, time80_array, time90_array, jitter_array, p1b_spe_array = initialize_arrays_2()
    file_path_calc, file_path_shift, file_path_shift_d1b, file_path_not_spe = initialize_folders_2(dest_path)

    # Mean p1 values
    mean_fwhm, mean_charge, mean_fall1090, mean_amplitude = mean_values(7.57e-9, 1.43e-12, 1.68e-8, 0.00661)

    # Checks jitter times
    print('Reading files...')
    for i in range(start, end + 1):
        if os.path.isfile(str(file_path_calc / 'D1--waveforms--%05d.txt') % i):
            myfile = open(str(file_path_calc / 'D1--waveforms--%05d.txt') % i, 'r')     # Opens file with calculations
            possibility = check_jitter(myfile)
            myfile.close()

            # If jitter times are unreasonable, adds file number to a list
            if possibility == 'no':
                jitter_array = np.append(jitter_array, int(i))

    for i in range(start, end + 1):
        if os.path.isfile(str(file_path_not_spe / 'D1--waveforms--%05d.txt') % i):
            pass
        elif os.path.isfile(str(file_path_shift_d1b / 'D1--waveforms--%05d.txt') % i):
            pass
        else:
            if os.path.isfile(str(file_path_shift / 'D1--waveforms--%05d.txt') % i):
                t, v, hdr = rw(str(file_path_shift / 'D1--waveforms--%05d.txt') % i, nhdr)      # Reads waveform file

                t1, t2, charge, amp, fwhm, rise1090, rise2080, fall1090, fall2080, j10, j20, j80, j90 = \
                    read_calc(str(file_path_calc / 'D1--waveforms--%05d.txt') % i)

                # If FWHM, charge, or 10-90 fall time is over twice the mean value, waveform is not spe
                possibility2 = check_vals(fwhm, charge, fall1090, amp, mean_fwhm, mean_charge, mean_fall1090,
                                          mean_amplitude)

                if possibility2 == 'no':
                    print('File #%05d is not spe' % i)
                    ww(t, v, str(file_path_not_spe / 'D1--waveforms--%05d.txt') % i, hdr)
                else:
                    p1b_sort(i, nhdr, jitter_array, p1b_spe_array, file_path_shift, file_path_shift_d1b,
                             file_path_not_spe)

    for i in range(start, end + 1):
        if i in p1b_spe_array:      # If a waveform is spe as sorted by p1b, its calculations are added to arrays
            print("Reading calculations from shifted file #%05d" % i)

            t1, t2, charge, amp, fwhm, rise1090, rise2080, fall1090, fall2080, j10, j20, j80, j90 = \
                read_calc(str(file_path_calc / 'D1--waveforms--%05d.txt') % i)
            charge_array, amplitude_array, fwhm_array, rise1090_array, rise2080_array, fall1090_array, fall2080_array, \
                time10_array, time20_array, time80_array, time90_array = \
                p1b_calc_arrays(charge, amp, fwhm, rise1090, rise2080, fall1090, fall2080, j10, j20, j80, j90,
                                charge_array, amplitude_array, fwhm_array, rise1090_array, rise2080_array,
                                fall1090_array, fall2080_array, time10_array, time20_array, time80_array, time90_array)

    # Plots histograms of charge, amplitude, FWHM, 10-90 & 20-80 rise times, 10-90 & 20-80 fall times, and 10%, 20%, 80%
    # & 90% jitter
    p1_hist(charge_array, amplitude_array, fwhm_array, rise1090_array, rise2080_array, fall1090_array, fall2080_array,
            time10_array, time20_array, time80_array, time90_array, dest_path, 100, 'd1b')


if __name__ == '__main__':
    data = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/waveforms/full_bdw_no_nf/d1')
    import argparse
    parser = argparse.ArgumentParser(prog="p1b", description="Removing outliers from data set")
    parser.add_argument("--start", type=int, help='file number to begin at', default=00000)
    parser.add_argument("--end", type=int, help='file number to end at', default=99999)
    parser.add_argument("--nhdr", type=int, help='number of header lines to skip', default=5)
    parser.add_argument("--dest_path", type=str, help='folder to read from', default=data)
    args = parser.parse_args()

    p1b(args.start, args.end, args.dest_path, args.nhdr)