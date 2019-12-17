from p3_functions import *


# Downsamples and digitizes double spe waveforms, then calculates charge, amplitude, and FWHM
def double_spe_studies(date, filter_band, nhdr, delay_folder, fsps, fsps_new, noise, r):
    gen_path, save_path, data_path, dest_path, filt_path1, filt_path2, filt_path4, filt_path8 = \
        initialize_folders(date, filter_band)
    make_folders(dest_path, filt_path1, filt_path2, filt_path4, filt_path8, fsps_new, delay_folder)

    single_file_array, double_file_array = initial_arrays(Path(Path(dest_path / 'rt_1_single_2') / 'raw'),
                                                          Path(filt_path1 / 'raw' / delay_folder))

    single_file_array_2, double_file_array_2 = initial_arrays_2(Path(data_path / 'rt_1_single_2'),
                                                                Path(data_path / 'rt_1_double' / delay_folder))

    single_file_array = copy_s_waveforms(single_file_array_2, single_file_array, data_path, dest_path, nhdr)
    double_file_array = copy_d_waveforms(double_file_array_2, double_file_array, data_path, filt_path1, filt_path2,
                                         filt_path4, filt_path8, delay_folder, nhdr)

    down_dig(single_file_array, double_file_array, filt_path1, filt_path2, filt_path4, filt_path8, dest_path,
             delay_folder, fsps, fsps_new, noise, nhdr)

    # Creates single spe arrays for charge, amplitude, and FWHM
    t1_array_s_1, t2_array_s_1, charge_array_s_1, amplitude_array_s_1, fwhm_array_s_1 = \
        make_arrays_s(Path(dest_path / 'rt_1_single_2' / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps')),
                      dest_path, single_file_array, nhdr, r, fsps_new, 'rt_1')
    t1_array_s_2, t2_array_s_2, charge_array_s_2, amplitude_array_s_2, fwhm_array_s_2 = \
        make_arrays_s(Path(dest_path / 'rt_2_single_2' / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps')),
                      dest_path, single_file_array, nhdr, r, fsps_new, 'rt_2')
    t1_array_s_4, t2_array_s_4, charge_array_s_4, amplitude_array_s_4, fwhm_array_s_4 = \
        make_arrays_s(Path(dest_path / 'rt_4_single_2' / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps')),
                      dest_path, single_file_array, nhdr, r, fsps_new, 'rt_4')
    t1_array_s_8, t2_array_s_8, charge_array_s_8, amplitude_array_s_8, fwhm_array_s_8 = \
        make_arrays_s(Path(dest_path / 'rt_8_single_2' / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps')),
                      dest_path, single_file_array, nhdr, r, fsps_new, 'rt_8')

    # Creates double spe arrays for charge, amplitude, and FWHM
    t1_array_d_1, t2_array_d_1, charge_array_d_1, amplitude_array_d_1, fwhm_array_d_1 = \
        make_arrays_d(Path(dest_path / 'rt_1_double' / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') /
                           delay_folder), dest_path, delay_folder, double_file_array, nhdr, r, fsps_new, 'rt_1')
    t1_array_d_2, t2_array_d_2, charge_array_d_2, amplitude_array_d_2, fwhm_array_d_2 = \
        make_arrays_d(Path(dest_path / 'rt_2_double' / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') /
                           delay_folder), dest_path, delay_folder, double_file_array, nhdr, r, fsps_new, 'rt_2')
    t1_array_d_4, t2_array_d_4, charge_array_d_4, amplitude_array_d_4, fwhm_array_d_4 = \
        make_arrays_d(Path(dest_path / 'rt_4_double' / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') /
                           delay_folder), dest_path, delay_folder, double_file_array, nhdr, r, fsps_new, 'rt_4')
    t1_array_d_8, t2_array_d_8, charge_array_d_8, amplitude_array_d_8, fwhm_array_d_8 = \
        make_arrays_d(Path(dest_path / 'rt_8_double' / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') /
                           delay_folder), dest_path, delay_folder, double_file_array, nhdr, r, fsps_new, 'rt_8')

    # Creates single and double spe histograms for charge, amplitude, and FWHM
    p3_hist(dest_path, delay_folder, charge_array_s_1, charge_array_s_2, charge_array_s_4, charge_array_s_8,
            amplitude_array_s_1, amplitude_array_s_2, amplitude_array_s_4, amplitude_array_s_8, fwhm_array_s_1,
            fwhm_array_s_2, fwhm_array_s_4, fwhm_array_s_8, charge_array_d_1, charge_array_d_2, charge_array_d_4,
            charge_array_d_8, amplitude_array_d_1, amplitude_array_d_2, amplitude_array_d_4, amplitude_array_d_8,
            fwhm_array_d_1, fwhm_array_d_2, fwhm_array_d_4, fwhm_array_d_8, fsps_new)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog="p3_double_studies", description="Creating double spe D3")
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