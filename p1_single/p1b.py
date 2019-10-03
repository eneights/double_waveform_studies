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
                if charge > 2 * mean_charge and (fwhm > 2 * mean_fwhm or fall1090 > 2 * mean_fall1090 or amp > 2 *
                                                 mean_amplitude):
                    print('File #%05d is not spe' % i)
                    ww(t, v, str(file_path_not_spe / 'D1--waveforms--%05d.txt') % i, hdr)
                else:
                    val = 0
                    for j in range(len(jitter_array)):
                        if jitter_array[j] == i:
                            val = 1
                    if val == 1:     # If a file had unreasonable jitter times, plots waveform for user to sort manually
                        t, v, hdr = rw(str(file_path_shift / 'D1--waveforms--%05d.txt') % i, nhdr)  # Reads waveform
                        print('Displaying file #%05d' % i)
                        plt.figure()
                        plt.plot(t, v)
                        plt.title('File #%05d' % i)
                        plt.xlabel('Time (s)')
                        plt.ylabel('Voltage (V)')
                        plt.show()
                        spe_check = 'pre-loop initialization'
                        while spe_check != 'y' and spe_check != 'n' and spe_check != 'u':
                            spe_check = input('Is this a normal SPE? "y" or "n"\n')
                        if spe_check == 'y':
                            print('File #%05d is spe' % i)
                            ww(t, v, str(file_path_shift_d1b / 'D1--waveforms--%05d.txt') % i, hdr)
                            p1b_spe_array = np.append(p1b_spe_array, i)     # File numbers of spes are added to an array
                        elif spe_check == 'n':
                            print('File #%05d is not spe' % i)
                            ww(t, v, str(file_path_not_spe / 'D1--waveforms--%05d.txt') % i, hdr)
                        plt.close()
                    else:       # If a file did not have unreasonable jitter times, it is spe
                        t, v, hdr = rw(str(file_path_shift / 'D1--waveforms--%05d.txt') % i, nhdr)
                        ww(t, v, str(file_path_shift_d1b / 'D1--waveforms--%05d.txt') % i, hdr)
                        p1b_spe_array = np.append(p1b_spe_array, i)         # File numbers of spes are added to an array

    for i in range(start, end + 1):
        if i in p1b_spe_array:      # If a waveform is spe as sorted by p1b, its calculations are added to arrays
            print("Reading calculations from shifted file #%05d" % i)
            myfile = open(str(file_path_calc / 'D1--waveforms--%05d.txt') % i, 'r')     # Opens file with calculations
            csv_reader = csv.reader(myfile)
            file_array = np.array([])
            for row in csv_reader:          # Creates array with calculation data
                file_array = np.append(file_array, float(row[1]))
            myfile.close()
            charge = file_array[2]
            amplitude = file_array[3]
            fwhm = file_array[4]
            rise1090 = file_array[5]
            rise2080 = file_array[6]
            fall1090 = file_array[7]
            fall2080 = file_array[8]
            time10 = file_array[9]
            time20 = file_array[10]
            time80 = file_array[11]
            time90 = file_array[12]

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

    # Plots histograms of charge, amplitude, FWHM, 10-90 & 20-80 rise times, 10-90 & 20-80 fall times, and 10%, 20%, 80%
    # & 90% jitter
    p1_hist(charge_array, amplitude_array, fwhm_array, rise1090_array, rise2080_array, fall1090_array, fall2080_array,
            time10_array, time20_array, time80_array, time90_array, dest_path, 100, 'p1b')


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