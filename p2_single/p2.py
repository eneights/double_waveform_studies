from p2_functions import *


# Creates data sets of spe waveforms with 2x, 4x, and 8x the initial rise times
def p2(start, end, date, date_time, filter_band, nhdr, fsps, r, pmt_hv, gain, offset, trig_delay, amp, band, nfilter):
    gen_path, save_path, data_path, initial_data, dest_path, filt_path1, filt_path2, filt_path4, filt_path8 =\
        initialize_folders(date, filter_band)
    make_folders(dest_path, filt_path1, filt_path2, filt_path4, filt_path8)

    print('Transferring files to rt_1 folder...')
    for i in range(start, end + 1):
        if os.path.isfile(Path(str((initial_data / 'D1--waveforms--%05d.txt')) % i)):
            if os.path.isfile(str(filt_path1 / 'D2--waveforms--%05d.txt') % i):
                pass
            else:
                t, v, hdr = rw(Path(str((initial_data / 'D1--waveforms--%05d.txt')) % i), nhdr)
                ww(t, v, str(filt_path1 / 'D2--waveforms--%05d.txt') % i, hdr)

    print('Calculating taus...')
    # Uses average spe waveform to calculate tau to use in lowpass filter for 2x rise time
    average_file = str(data_path / 'hist_data' / 'avg_waveform_d1b.txt')
    tau_2, tau_4, tau_8, v1, v2, v4, v8 = taus(average_file, fsps, nhdr)

    v_gain, v2_gain, v4_gain, v8_gain, factor2, factor4, factor8 = calc_gain(v1, v2, v4, v8)

    avg_shapings(average_file, dest_path, v_gain, v2_gain, v4_gain, v8_gain, tau_2, tau_4, tau_8, nhdr)

    # For each spe waveform file, calculates and saves waveforms with 1x, 2x, 4x, and 8x the rise time
    for i in range(start, end + 1):
        save_name1 = str(filt_path1 / 'D2--waveforms--%05d.txt') % i
        save_name2 = str(filt_path2 / 'D2--waveforms--%05d.txt') % i
        save_name4 = str(filt_path4 / 'D2--waveforms--%05d.txt') % i
        save_name8 = str(filt_path8 / 'D2--waveforms--%05d.txt') % i

        shaping(save_name1, save_name2, save_name4, save_name8, i, tau_2, tau_4, tau_8, factor2, factor4,
                factor8, fsps, nhdr)

    # Plots average waveform for 1x rise time
    print('Calculating rt_1 average waveform...')
    average_waveform(start, end, dest_path, 'rt_1', nhdr)

    # Plots average waveform for 2x rise time
    print('Calculating rt_2 average waveform...')
    average_waveform(start, end, dest_path, 'rt_2', nhdr)

    # Plots average waveform for 4x rise time
    print('Calculating rt_4 average waveform...')
    average_waveform(start, end, dest_path, 'rt_4', nhdr)

    # Plots average waveform for 8x rise time
    print('Calculating rt_8 average waveform...')
    average_waveform(start, end, dest_path, 'rt_8', nhdr)

    # Calculates 10-90 rise times for each waveform and puts them into arrays
    print('Doing calculations...')
    rt_1_array, rt_2_array, rt_4_array, rt_8_array = make_arrays(filt_path1, filt_path2, filt_path4, filt_path8,
                                                                 dest_path, start, end, nhdr)

    # Creates histograms of 10-90 rise times for 1x, 2x, 4x, and 8x the initial rise time
    print('Creating histograms...')
    p2_hist(rt_1_array, rt_2_array, rt_4_array, rt_8_array, dest_path, 100)

    # Writes info file
    info_file(date_time, data_path, dest_path, pmt_hv, gain, offset, trig_delay, amp, fsps, band, nfilter, r)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog="p2", description="Creating D2")
    parser.add_argument("--start", type=int, help='file number to begin at (default=0)', default=0)
    parser.add_argument("--end", type=int, help='file number to end at (default=99999)', default=99999)
    parser.add_argument("--date", type=int, help='date of data acquisition (YEARMMDD)')
    parser.add_argument("--date_time", type=str, help='date & time of d1 processing')
    parser.add_argument("--fil_band", type=str, help='folder name for data')
    parser.add_argument("--nhdr", type=int, help='number of header lines to skip in raw file (default=5)', default=5)
    parser.add_argument("--fsps", type=float, help='samples per second (Hz) (suggested=20000000000.)')
    parser.add_argument("--r", type=int, help='resistance in ohms (suggested=50)')
    parser.add_argument("--pmt_hv", type=int, help='voltage of PMT (V) (suggested=1800)')
    parser.add_argument("--gain", type=int, help='gain of PMT (suggested=1e7)')
    parser.add_argument("--offset", type=int, help='offset of pulse generator (suggested=0)')
    parser.add_argument("--trig_delay", type=float, help='delay of pulse generator trigger (ns) (suggested=9.)')
    parser.add_argument("--amp", type=float, help='amplitude of pulse generator (V) (suggested=3.5)')
    parser.add_argument("--band", type=str, help='bandwidth of oscilloscope (Hz)')
    parser.add_argument("--nfilter", type=float, help='noise filter on oscilloscope (bits)')
    parser.add_argument("--info_file", type=str, help='path to d1 info file')
    args = parser.parse_args()

    if not args.info_file:
        if not (args.date or args.date_time or args.fil_band or args.fsps or args.r or args.pmt_hv or
                args.gain or args.offset or args.trig_delay or args.amp or args.band or args.nfilter):
            print('Error: Must provide an info file or all other arguments')
        else:
            p2(args.start, args.end, args.date, args.date_time, args.fil_band, args.nhdr, args.fsps, args.r,
               args.pmt_hv, args.gain, args.offset, args.trig_delay, args.amp, args.band, args.nfilter)
    else:
        myfile = open(args.info_file, 'r')
        csv_reader = csv.reader(myfile)
        info_array = np.array([])
        path_array = np.array([])
        for row in csv_reader:
            info_array = np.append(info_array, row[1])
        i_date_time = info_array[1]
        i_path = info_array[3]
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

        p2(args.start, args.end, i_date, i_date_time, i_fil_band, args.nhdr, i_fsps, i_r, i_pmt_hv, i_gain, i_offset,
           i_trig_delay, i_amp, i_band, i_nfilter)

        myfile.close()