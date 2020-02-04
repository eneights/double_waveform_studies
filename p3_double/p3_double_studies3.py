from p3_functions import *


def double_spe_studies_3(date, filter_band, fsps_new):
    gen_path, save_path, dest_path, calc_single, calc_double = initialize_folders3(date, filter_band)

    amp_1, charge_1, fwhm_1 = cutoff_vals(fsps_new, 'rt_1')
    amp_2, charge_2, fwhm_2 = cutoff_vals(fsps_new, 'rt_2')
    amp_4, charge_4, fwhm_4 = cutoff_vals(fsps_new, 'rt_4')
    amp_8, charge_8, fwhm_8 = cutoff_vals(fsps_new, 'rt_8')

    print('Calculating single files...')
    true_spes_per_1, false_mpes_per_1 = sort_single(amp_1, charge_1, fwhm_1, calc_single, fsps_new, 'rt_1')
    true_spes_per_2, false_mpes_per_2 = sort_single(amp_2, charge_2, fwhm_2, calc_single, fsps_new, 'rt_2')
    true_spes_per_4, false_mpes_per_4 = sort_single(amp_4, charge_4, fwhm_4, calc_single, fsps_new, 'rt_4')
    true_spes_per_8, false_mpes_per_8 = sort_single(amp_8, charge_8, fwhm_8, calc_single, fsps_new, 'rt_8')

    print('Calculating double files...')
    true_mpes_per_1_no, false_spes_per_1_no = sort_double(amp_1, charge_1, fwhm_1, calc_double, fsps_new, 'no_delay',
                                                          'rt_1')
    true_mpes_per_2_no, false_spes_per_2_no = sort_double(amp_2, charge_2, fwhm_2, calc_double, fsps_new, 'no_delay',
                                                          'rt_2')
    true_mpes_per_4_no, false_spes_per_4_no = sort_double(amp_4, charge_4, fwhm_4, calc_double, fsps_new, 'no_delay',
                                                          'rt_4')
    true_mpes_per_8_no, false_spes_per_8_no = sort_double(amp_8, charge_8, fwhm_8, calc_double, fsps_new, 'no_delay',
                                                          'rt_8')

    true_mpes_per_1_05x, false_spes_per_1_05x = sort_double(amp_1, charge_1, fwhm_1, calc_double, fsps_new, '0.5x_rt',
                                                            'rt_1')
    true_mpes_per_2_05x, false_spes_per_2_05x = sort_double(amp_2, charge_2, fwhm_2, calc_double, fsps_new, '0.5x_rt',
                                                            'rt_2')
    true_mpes_per_4_05x, false_spes_per_4_05x = sort_double(amp_4, charge_4, fwhm_4, calc_double, fsps_new, '0.5x_rt',
                                                            'rt_4')
    true_mpes_per_8_05x, false_spes_per_8_05x = sort_double(amp_8, charge_8, fwhm_8, calc_double, fsps_new, '0.5x_rt',
                                                            'rt_8')

    true_mpes_per_1_1x, false_spes_per_1_1x = sort_double(amp_1, charge_1, fwhm_1, calc_double, fsps_new, '1x_rt',
                                                          'rt_1')
    true_mpes_per_2_1x, false_spes_per_2_1x = sort_double(amp_2, charge_2, fwhm_2, calc_double, fsps_new, '1x_rt',
                                                          'rt_2')
    true_mpes_per_4_1x, false_spes_per_4_1x = sort_double(amp_4, charge_4, fwhm_4, calc_double, fsps_new, '1x_rt',
                                                          'rt_4')
    true_mpes_per_8_1x, false_spes_per_8_1x = sort_double(amp_8, charge_8, fwhm_8, calc_double, fsps_new, '1x_rt',
                                                          'rt_8')

    true_mpes_per_1_15x, false_spes_per_1_15x = sort_double(amp_1, charge_1, fwhm_1, calc_double, fsps_new, '1.5x_rt',
                                                            'rt_1')
    true_mpes_per_2_15x, false_spes_per_2_15x = sort_double(amp_2, charge_2, fwhm_2, calc_double, fsps_new, '1.5x_rt',
                                                            'rt_2')
    true_mpes_per_4_15x, false_spes_per_4_15x = sort_double(amp_4, charge_4, fwhm_4, calc_double, fsps_new, '1.5x_rt',
                                                            'rt_4')
    true_mpes_per_8_15x, false_spes_per_8_15x = sort_double(amp_8, charge_8, fwhm_8, calc_double, fsps_new, '1.5x_rt',
                                                            'rt_8')

    true_mpes_per_1_2x, false_spes_per_1_2x = sort_double(amp_1, charge_1, fwhm_1, calc_double, fsps_new, '2x_rt',
                                                          'rt_1')
    true_mpes_per_2_2x, false_spes_per_2_2x = sort_double(amp_2, charge_2, fwhm_2, calc_double, fsps_new, '2x_rt',
                                                          'rt_2')
    true_mpes_per_4_2x, false_spes_per_4_2x = sort_double(amp_4, charge_4, fwhm_4, calc_double, fsps_new, '2x_rt',
                                                          'rt_4')
    true_mpes_per_8_2x, false_spes_per_8_2x = sort_double(amp_8, charge_8, fwhm_8, calc_double, fsps_new, '2x_rt',
                                                          'rt_8')

    true_mpes_per_1_25x, false_spes_per_1_25x = sort_double(amp_1, charge_1, fwhm_1, calc_double, fsps_new, '2.5x_rt',
                                                            'rt_1')
    true_mpes_per_2_25x, false_spes_per_2_25x = sort_double(amp_2, charge_2, fwhm_2, calc_double, fsps_new, '2.5x_rt',
                                                            'rt_2')
    true_mpes_per_4_25x, false_spes_per_4_25x = sort_double(amp_4, charge_4, fwhm_4, calc_double, fsps_new, '2.5x_rt',
                                                            'rt_4')
    true_mpes_per_8_25x, false_spes_per_8_25x = sort_double(amp_8, charge_8, fwhm_8, calc_double, fsps_new, '2.5x_rt',
                                                            'rt_8')

    true_mpes_per_1_3x, false_spes_per_1_3x = sort_double(amp_1, charge_1, fwhm_1, calc_double, fsps_new, '3x_rt',
                                                          'rt_1')
    true_mpes_per_2_3x, false_spes_per_2_3x = sort_double(amp_2, charge_2, fwhm_2, calc_double, fsps_new, '3x_rt',
                                                          'rt_2')
    true_mpes_per_4_3x, false_spes_per_4_3x = sort_double(amp_4, charge_4, fwhm_4, calc_double, fsps_new, '3x_rt',
                                                          'rt_4')
    true_mpes_per_8_3x, false_spes_per_8_3x = sort_double(amp_8, charge_8, fwhm_8, calc_double, fsps_new, '3x_rt',
                                                          'rt_8')

    true_mpes_per_1_35x, false_spes_per_1_35x = sort_double(amp_1, charge_1, fwhm_1, calc_double, fsps_new, '3.5x_rt',
                                                            'rt_1')
    true_mpes_per_2_35x, false_spes_per_2_35x = sort_double(amp_2, charge_2, fwhm_2, calc_double, fsps_new, '3.5x_rt',
                                                            'rt_2')
    true_mpes_per_4_35x, false_spes_per_4_35x = sort_double(amp_4, charge_4, fwhm_4, calc_double, fsps_new, '3.5x_rt',
                                                            'rt_4')
    true_mpes_per_8_35x, false_spes_per_8_35x = sort_double(amp_8, charge_8, fwhm_8, calc_double, fsps_new, '3.5x_rt',
                                                            'rt_8')

    true_mpes_per_1_4x, false_spes_per_1_4x = sort_double(amp_1, charge_1, fwhm_1, calc_double, fsps_new, '4x_rt',
                                                          'rt_1')
    true_mpes_per_2_4x, false_spes_per_2_4x = sort_double(amp_2, charge_2, fwhm_2, calc_double, fsps_new, '4x_rt',
                                                          'rt_2')
    true_mpes_per_4_4x, false_spes_per_4_4x = sort_double(amp_4, charge_4, fwhm_4, calc_double, fsps_new, '4x_rt',
                                                          'rt_4')
    true_mpes_per_8_4x, false_spes_per_8_4x = sort_double(amp_8, charge_8, fwhm_8, calc_double, fsps_new, '4x_rt',
                                                          'rt_8')

    true_mpes_per_1_45x, false_spes_per_1_45x = sort_double(amp_1, charge_1, fwhm_1, calc_double, fsps_new, '4.5x_rt',
                                                            'rt_1')
    true_mpes_per_2_45x, false_spes_per_2_45x = sort_double(amp_2, charge_2, fwhm_2, calc_double, fsps_new, '4.5x_rt',
                                                            'rt_2')
    true_mpes_per_4_45x, false_spes_per_4_45x = sort_double(amp_4, charge_4, fwhm_4, calc_double, fsps_new, '4.5x_rt',
                                                            'rt_4')
    true_mpes_per_8_45x, false_spes_per_8_45x = sort_double(amp_8, charge_8, fwhm_8, calc_double, fsps_new, '4.5x_rt',
                                                            'rt_8')

    true_mpes_per_1_5x, false_spes_per_1_5x = sort_double(amp_1, charge_1, fwhm_1, calc_double, fsps_new, '5x_rt',
                                                          'rt_1')
    true_mpes_per_2_5x, false_spes_per_2_5x = sort_double(amp_2, charge_2, fwhm_2, calc_double, fsps_new, '5x_rt',
                                                          'rt_2')
    true_mpes_per_4_5x, false_spes_per_4_5x = sort_double(amp_4, charge_4, fwhm_4, calc_double, fsps_new, '5x_rt',
                                                          'rt_4')
    true_mpes_per_8_5x, false_spes_per_8_5x = sort_double(amp_8, charge_8, fwhm_8, calc_double, fsps_new, '5x_rt',
                                                          'rt_8')

    true_mpes_per_1_55x, false_spes_per_1_55x = sort_double(amp_1, charge_1, fwhm_1, calc_double, fsps_new, '5.5x_rt',
                                                            'rt_1')
    true_mpes_per_2_55x, false_spes_per_2_55x = sort_double(amp_2, charge_2, fwhm_2, calc_double, fsps_new, '5.5x_rt',
                                                            'rt_2')
    true_mpes_per_4_55x, false_spes_per_4_55x = sort_double(amp_4, charge_4, fwhm_4, calc_double, fsps_new, '5.5x_rt',
                                                            'rt_4')
    true_mpes_per_8_55x, false_spes_per_8_55x = sort_double(amp_8, charge_8, fwhm_8, calc_double, fsps_new, '5.5x_rt',
                                                            'rt_8')

    true_mpes_per_1_6x, false_spes_per_1_6x = sort_double(amp_1, charge_1, fwhm_1, calc_double, fsps_new, '6x_rt',
                                                          'rt_1')
    true_mpes_per_2_6x, false_spes_per_2_6x = sort_double(amp_2, charge_2, fwhm_2, calc_double, fsps_new, '6x_rt',
                                                          'rt_2')
    true_mpes_per_4_6x, false_spes_per_4_6x = sort_double(amp_4, charge_4, fwhm_4, calc_double, fsps_new, '6x_rt',
                                                          'rt_4')
    true_mpes_per_8_6x, false_spes_per_8_6x = sort_double(amp_8, charge_8, fwhm_8, calc_double, fsps_new, '6x_rt',
                                                          'rt_8')

    true_mpes_per_1_40, false_spes_per_1_40 = sort_double(amp_1, charge_1, fwhm_1, calc_double, fsps_new, '40_ns',
                                                          'rt_1')
    true_mpes_per_2_40, false_spes_per_2_40 = sort_double(amp_2, charge_2, fwhm_2, calc_double, fsps_new, '40_ns',
                                                          'rt_2')
    true_mpes_per_4_40, false_spes_per_4_40 = sort_double(amp_4, charge_4, fwhm_4, calc_double, fsps_new, '40_ns',
                                                          'rt_4')
    true_mpes_per_8_40, false_spes_per_8_40 = sort_double(amp_8, charge_8, fwhm_8, calc_double, fsps_new, '40_ns',
                                                          'rt_8')

    true_mpes_per_1_80, false_spes_per_1_80 = sort_double(amp_1, charge_1, fwhm_1, calc_double, fsps_new, '80_ns',
                                                          'rt_1')
    true_mpes_per_2_80, false_spes_per_2_80 = sort_double(amp_2, charge_2, fwhm_2, calc_double, fsps_new, '80_ns',
                                                          'rt_2')
    true_mpes_per_4_80, false_spes_per_4_80 = sort_double(amp_4, charge_4, fwhm_4, calc_double, fsps_new, '80_ns',
                                                          'rt_4')
    true_mpes_per_8_80, false_spes_per_8_80 = sort_double(amp_8, charge_8, fwhm_8, calc_double, fsps_new, '80_ns',
                                                          'rt_8')

    print(str(str(int(fsps_new / 1e6)) + ' Msps (No Shaping)'))
    print('False MPEs: ' + str(false_mpes_per_1) + '%')
    print('False SPEs (no delay): ' + str(false_spes_per_1_no) + '%')
    print('False SPEs (0.5x rt delay): ' + str(false_spes_per_1_05x) + '%')
    print('False SPEs (1x rt delay): ' + str(false_spes_per_1_1x) + '%')
    print('False SPEs (1.5x rt delay): ' + str(false_spes_per_1_15x) + '%')
    print('False SPEs (2x rt delay): ' + str(false_spes_per_1_2x) + '%')
    print('False SPEs (2.5x rt delay): ' + str(false_spes_per_1_25x) + '%')
    print('False SPEs (3x rt delay): ' + str(false_spes_per_1_3x) + '%')
    print('False SPEs (3.5x rt delay): ' + str(false_spes_per_1_35x) + '%')
    print('False SPEs (4x rt delay): ' + str(false_spes_per_1_4x) + '%')
    print('False SPEs (4.5x rt delay): ' + str(false_spes_per_1_45x) + '%')
    print('False SPEs (5x rt delay): ' + str(false_spes_per_1_5x) + '%')
    print('False SPEs (5.5x rt delay): ' + str(false_spes_per_1_55x) + '%')
    print('False SPEs (6x rt delay): ' + str(false_spes_per_1_6x) + '%')
    print('False SPEs (40 ns delay): ' + str(false_spes_per_1_40) + '%')
    print('False SPEs (80 ns delay): ' + str(false_spes_per_1_80) + '%\n')

    print(str(str(int(fsps_new / 1e6)) + ' Msps (2x Shaping)'))
    print('False MPEs: ' + str(false_mpes_per_2) + '%')
    print('False SPEs (no delay): ' + str(false_spes_per_2_no) + '%')
    print('False SPEs (0.5x rt delay): ' + str(false_spes_per_2_05x) + '%')
    print('False SPEs (1x rt delay): ' + str(false_spes_per_2_1x) + '%')
    print('False SPEs (1.5x rt delay): ' + str(false_spes_per_2_15x) + '%')
    print('False SPEs (2x rt delay): ' + str(false_spes_per_2_2x) + '%')
    print('False SPEs (2.5x rt delay): ' + str(false_spes_per_2_25x) + '%')
    print('False SPEs (3x rt delay): ' + str(false_spes_per_2_3x) + '%')
    print('False SPEs (3.5x rt delay): ' + str(false_spes_per_2_35x) + '%')
    print('False SPEs (4x rt delay): ' + str(false_spes_per_2_4x) + '%')
    print('False SPEs (4.5x rt delay): ' + str(false_spes_per_2_45x) + '%')
    print('False SPEs (5x rt delay): ' + str(false_spes_per_2_5x) + '%')
    print('False SPEs (5.5x rt delay): ' + str(false_spes_per_2_55x) + '%')
    print('False SPEs (6x rt delay): ' + str(false_spes_per_2_6x) + '%')
    print('False SPEs (40 ns delay): ' + str(false_spes_per_2_40) + '%')
    print('False SPEs (80 ns delay): ' + str(false_spes_per_2_80) + '%\n')

    print(str(str(int(fsps_new / 1e6)) + ' Msps (4x Shaping)'))
    print('False MPEs: ' + str(false_mpes_per_4) + '%')
    print('False SPEs (no delay): ' + str(false_spes_per_4_no) + '%')
    print('False SPEs (0.5x rt delay): ' + str(false_spes_per_4_05x) + '%')
    print('False SPEs (1x rt delay): ' + str(false_spes_per_4_1x) + '%')
    print('False SPEs (1.5x rt delay): ' + str(false_spes_per_4_15x) + '%')
    print('False SPEs (2x rt delay): ' + str(false_spes_per_4_2x) + '%')
    print('False SPEs (2.5x rt delay): ' + str(false_spes_per_4_25x) + '%')
    print('False SPEs (3x rt delay): ' + str(false_spes_per_4_3x) + '%')
    print('False SPEs (3.5x rt delay): ' + str(false_spes_per_4_35x) + '%')
    print('False SPEs (4x rt delay): ' + str(false_spes_per_4_4x) + '%')
    print('False SPEs (4.5x rt delay): ' + str(false_spes_per_4_45x) + '%')
    print('False SPEs (5x rt delay): ' + str(false_spes_per_4_5x) + '%')
    print('False SPEs (5.5x rt delay): ' + str(false_spes_per_4_55x) + '%')
    print('False SPEs (6x rt delay): ' + str(false_spes_per_4_6x) + '%')
    print('False SPEs (40 ns delay): ' + str(false_spes_per_4_40) + '%')
    print('False SPEs (80 ns delay): ' + str(false_spes_per_4_80) + '%\n')

    print(str(str(int(fsps_new / 1e6)) + ' Msps (8x Shaping)'))
    print('False MPEs: ' + str(false_mpes_per_8) + '%')
    print('False SPEs (no delay): ' + str(false_spes_per_8_no) + '%')
    print('False SPEs (0.5x rt delay): ' + str(false_spes_per_8_05x) + '%')
    print('False SPEs (1x rt delay): ' + str(false_spes_per_8_1x) + '%')
    print('False SPEs (1.5x rt delay): ' + str(false_spes_per_8_15x) + '%')
    print('False SPEs (2x rt delay): ' + str(false_spes_per_8_2x) + '%')
    print('False SPEs (2.5x rt delay): ' + str(false_spes_per_8_25x) + '%')
    print('False SPEs (3x rt delay): ' + str(false_spes_per_8_3x) + '%')
    print('False SPEs (3.5x rt delay): ' + str(false_spes_per_8_35x) + '%')
    print('False SPEs (4x rt delay): ' + str(false_spes_per_8_4x) + '%')
    print('False SPEs (4.5x rt delay): ' + str(false_spes_per_8_45x) + '%')
    print('False SPEs (5x rt delay): ' + str(false_spes_per_8_5x) + '%')
    print('False SPEs (5.5x rt delay): ' + str(false_spes_per_8_55x) + '%')
    print('False SPEs (6x rt delay): ' + str(false_spes_per_8_6x) + '%')
    print('False SPEs (40 ns delay): ' + str(false_spes_per_8_40) + '%')
    print('False SPEs (80 ns delay): ' + str(false_spes_per_8_80) + '%\n')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(prog="p3_double_studies3", description="Analyzing double spe data")
    parser.add_argument("--date", type=int, help='date of data acquisition (default=20190513)', default=20190513)
    parser.add_argument("--fil_band", type=str, help='folder name for data (default=full_bdw_no_nf)',
                        default='full_bdw_no_nf')
    parser.add_argument("--fsps_new", type=float, help='new samples per second (Hz) (default=500000000.)',
                        default=500000000.)
    args = parser.parse_args()

    double_spe_studies_3(args.date, args.fil_band, args.fsps_new)
