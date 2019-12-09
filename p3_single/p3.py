from p3_functions import *


# Downsamples and digitizes spe waveforms
def p3(start, end, date, date_time, filter_band, nhdr, fsps, r, pmt_hv, gain, offset, trig_delay, amp, band, nfilter,
       fsps_new, noise):
    gen_path, save_path, data_path, dest_path, filt_path1, filt_path2, filt_path4, filt_path8 = \
        initialize_folders(date, filter_band)
    make_folders(dest_path, filt_path1, filt_path2, filt_path4, filt_path8, fsps_new)

    # Copies waveforms with 1x, 2x, 4x, and 8x initial rise times to d3 folder
    for i in range(start, end + 1):
        transfer_files(data_path, filt_path1, filt_path2, filt_path4, filt_path8, i, nhdr)

    # Downsamples and digitizes waveforms
    down_dig(filt_path1, filt_path2, filt_path4, filt_path8, fsps, fsps_new, noise, start, end, nhdr)

    # Plots and saves average waveforms
    average_waveform(start, end, dest_path, 'rt_1', 'No Shaping', nhdr, fsps_new)
    average_waveform(start, end, dest_path, 'rt_2', '2x Rise Time Shaping', nhdr, fsps_new)
    average_waveform(start, end, dest_path, 'rt_4', '4x Rise Time Shaping', nhdr, fsps_new)
    average_waveform(start, end, dest_path, 'rt_8', '8x Rise Time Shaping', nhdr, fsps_new)

    # Writes info file
    info_file(date_time, data_path, dest_path, pmt_hv, gain, offset, trig_delay, amp, fsps, band, nfilter, r)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog="p3", description="Creating D3")
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
    parser.add_argument("--fsps_new", type=float, help='new samples per second (Hz) (suggested=500000000.)')
    parser.add_argument("--noise", type=float, help='noise to add (bits) (suggested=3.30)')
    parser.add_argument("--info_file", type=str, help='path to d2 info file')
    args = parser.parse_args()

    if not args.info_file:
        if not (args.date or args.date_time or args.fil_band or args.fsps or args.r or args.pmt_hv or
                args.gain or args.offset or args.trig_delay or args.amp or args.band or args.nfilter, args.fsps_new,
                args.noise):
            print('Error: Must provide an info file or all other arguments')
        else:
            p3(args.start, args.end, args.date, args.date_time, args.fil_band, args.nhdr, args.fsps, args.r,
               args.pmt_hv, args.gain, args.offset, args.trig_delay, args.amp, args.band, args.nfilter, args.fsps_new,
                args.noise)
    elif not (args.fsps_new or args.noise):
        print('Error: Must provide new fsps and noise level')
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

        p3(args.start, args.end, i_date, i_date_time, i_fil_band, args.nhdr, i_fsps, i_r, i_pmt_hv, i_gain, i_offset,
           i_trig_delay, i_amp, i_band, i_nfilter, args.fsps_new, args.noise)

        myfile.close()