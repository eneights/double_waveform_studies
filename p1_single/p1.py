from p1_functions import *


def p1(start, end, date, date_time, filter_band, nhdr, fsps, fc, numtaps, baseline, r, pmt_hv, gain, offset, trig_delay,
       amp, band, nfilter):
    gen_path, save_sort, data_sort, dest_path, data_shift, save_shift = initialize_folders(date, filter_band)

    # Separates spes and non-spes into different folders
    print('Sorting files...')
    for i in range(start, end + 1):
        p1_sort(i, nhdr, fsps, fc, numtaps, data_sort, save_sort, baseline)

    # Shifts spes so that when t = 0, v = 50% max and baseline = 0
    print('Shifting waveforms...')
    for i in range(start, end + 1):
        if os.path.isfile((data_shift / 'D1--waveforms--%05d.txt') % i):
            shift_waveform(i, nhdr, data_shift, save_shift)

    # Creates arrays of beginning & end times of spe waveform, time of end of spe, charge, amplitude, fwhm, 10-90 &
    # 20-80 rise times, 10-90 & 20-80 fall times, and 10%, 20%, 80% & 90% jitter
    t1_array, t2_array, charge_array, amplitude_array, fwhm_array, rise1090_array, rise2080_array, fall1090_array, \
        fall2080_array, time10_array, time20_array, time80_array, time90_array = make_arrays(save_shift, dest_path,
                                                                                             data_sort, start, end,
                                                                                             nhdr, r)

    # Plots histograms of charge, amplitude, FWHM, 10-90 & 20-80 rise times, 10-90 & 20-80 fall times, and 10%, 20%, 80%
    # & 90% jitter
    p1_hist(charge_array, amplitude_array, fwhm_array, rise1090_array, rise2080_array, fall1090_array, fall2080_array,
            time10_array, time20_array, time80_array, time90_array, dest_path, 100, 'p1')

    # Creates d1 info file
    info_file(date_time, data_sort, dest_path, pmt_hv, gain, offset, trig_delay, amp, fsps, band, nfilter, r)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog="p1", description="Creating D1")
    parser.add_argument("--start", type=int, help='file number to begin at (default=0)', default=0)
    parser.add_argument("--end", type=int, help='file number to end at (default=99999)', default=99999)
    parser.add_argument("--date", type=int, help='date of data acquisition (YEARMMDD)')
    parser.add_argument("--date_time", type=str, help='date & time of data acquisition')
    parser.add_argument("--fil_band", type=str, help='folder name for data')
    parser.add_argument("--nhdr", type=int, help='number of header lines to skip in raw file (default=5)', default=5)
    parser.add_argument("--fsps", type=float, help='samples per second (Hz) (suggested=20000000000.)')
    parser.add_argument("--fc", type=float, help='filter cutoff frequency (Hz) (default=250000000)', default=250000000.)
    parser.add_argument("--numtaps", type=int, help='filter order + 1 (default=51)', default=51)
    parser.add_argument("--baseline", type=float, help='baseline of data set (V) (suggested=0)')
    parser.add_argument("--r", type=int, help='resistance in ohms (suggested=50)')
    parser.add_argument("--pmt_hv", type=int, help='voltage of PMT (V) (suggested=1800)')
    parser.add_argument("--gain", type=int, help='gain of PMT (suggested=1e7)')
    parser.add_argument("--offset", type=int, help='offset of pulse generator (suggested=0)')
    parser.add_argument("--trig_delay", type=float, help='delay of pulse generator trigger (ns) (suggested=9.)')
    parser.add_argument("--amp", type=float, help='amplitude of pulse generator (V) (suggested=3.5)')
    parser.add_argument("--band", type=str, help='bandwidth of oscilloscope (Hz)')
    parser.add_argument("--nfilter", type=float, help='noise filter on oscilloscope (bits)')
    parser.add_argument("--info_file", type=str, help='path to d0 info file')
    args = parser.parse_args()

    if not args.info_file:
        if not (args.date or args.date_time or args.fil_band or args.fsps or args.baseline or args.r or args.pmt_hv or
                args.gain or args.offset or args.trig_delay or args.amp or args.band or args.nfilter):
            print('Error: Must provide an info file or all other arguments')
        else:
            p1(args.start, args.end, args.date, args.date_time, args.fil_band, args.nhdr, args.fsps, args.fc,
               args.numtaps, args.baseline, args.r, args.pmt_hv, args.gain, args.offset, args.trig_delay, args.amp,
               args.band, args.nfilter)
    else:
        myfile = open(args.info_file, 'r')
        i_date, i_date_time, i_fil_band, i_nhdr, i_fsps, i_baseline, i_r, i_pmt_hv, i_gain, i_offset, i_trig_delay, \
            i_amp, i_band, i_nfilter = read_info(myfile)

        p1(args.start, args.end, i_date, i_date_time, i_fil_band, i_nhdr, i_fsps, args.fc, args.numtaps,
           i_baseline, i_r, i_pmt_hv, i_gain, i_offset, i_trig_delay, i_amp, i_band, i_nfilter)

        myfile.close()