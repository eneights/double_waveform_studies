from p3_functions import *


def double_spe_studies_2(date, filter_band, fsps_new, shaping):
    gen_path, save_path, dest_path, hist_single, hist_double = initialize_folders2(date, filter_band)

    amp_sing, charge_sing, fwhm_sing, amp_doub_no_delay, amp_doub_05rt, amp_doub_1rt, amp_doub_15rt, amp_doub_2rt, \
    amp_doub_25rt, amp_doub_3rt, amp_doub_35rt, amp_doub_4rt, amp_doub_45rt, amp_doub_5rt, amp_doub_55rt, amp_doub_6rt,\
    amp_doub_40ns, amp_doub_80ns, charge_doub_no_delay, charge_doub_05rt, charge_doub_1rt, charge_doub_15rt, \
    charge_doub_2rt, charge_doub_25rt, charge_doub_3rt, charge_doub_35rt, charge_doub_4rt, charge_doub_45rt, \
    charge_doub_5rt, charge_doub_55rt, charge_doub_6rt, charge_doub_40ns, charge_doub_80ns, fwhm_doub_no_delay, \
    fwhm_doub_05rt, fwhm_doub_1rt, fwhm_doub_15rt, fwhm_doub_2rt, fwhm_doub_25rt, fwhm_doub_3rt, fwhm_doub_35rt, \
    fwhm_doub_4rt, fwhm_doub_45rt, fwhm_doub_5rt, fwhm_doub_55rt, fwhm_doub_6rt, fwhm_doub_40ns, fwhm_doub_80ns \
        = initialize_names(hist_single, hist_double, shaping, fsps_new)

    amp_sing_mean, charge_sing_mean, fwhm_sing_mean, amp_doub_no_delay_mean, amp_doub_05rt_mean, amp_doub_1rt_mean, \
    amp_doub_15rt_mean, amp_doub_2rt_mean, amp_doub_25rt_mean, amp_doub_3rt_mean, amp_doub_35rt_mean, \
    amp_doub_4rt_mean, amp_doub_45rt_mean, amp_doub_5rt_mean, amp_doub_55rt_mean, amp_doub_6rt_mean, \
    amp_doub_40ns_mean, amp_doub_80ns_mean, charge_doub_no_delay_mean, charge_doub_05rt_mean, charge_doub_1rt_mean, \
    charge_doub_15rt_mean, charge_doub_2rt_mean, charge_doub_25rt_mean, charge_doub_3rt_mean, charge_doub_35rt_mean, \
    charge_doub_4rt_mean, charge_doub_45rt_mean, charge_doub_5rt_mean, charge_doub_55rt_mean, charge_doub_6rt_mean, \
    charge_doub_40ns_mean, charge_doub_80ns_mean, fwhm_doub_no_delay_mean, fwhm_doub_05rt_mean, fwhm_doub_1rt_mean, \
    fwhm_doub_15rt_mean, fwhm_doub_2rt_mean, fwhm_doub_25rt_mean, fwhm_doub_3rt_mean, fwhm_doub_35rt_mean, \
    fwhm_doub_4rt_mean, fwhm_doub_45rt_mean, fwhm_doub_5rt_mean, fwhm_doub_55rt_mean, fwhm_doub_6rt_mean, \
    fwhm_doub_40ns_mean, fwhm_doub_80ns_mean, amp_sing_std, charge_sing_std, fwhm_sing_std, amp_doub_no_delay_std, \
    amp_doub_05rt_std, amp_doub_1rt_std, amp_doub_15rt_std, amp_doub_2rt_std, amp_doub_25rt_std, amp_doub_3rt_std, \
    amp_doub_35rt_std, amp_doub_4rt_std, amp_doub_45rt_std, amp_doub_5rt_std, amp_doub_55rt_std, amp_doub_6rt_std, \
    amp_doub_40ns_std, amp_doub_80ns_std, charge_doub_no_delay_std, charge_doub_05rt_std, charge_doub_1rt_std, \
    charge_doub_15rt_std, charge_doub_2rt_std, charge_doub_25rt_std, charge_doub_3rt_std, charge_doub_35rt_std, \
    charge_doub_4rt_std, charge_doub_45rt_std, charge_doub_5rt_std, charge_doub_55rt_std, charge_doub_6rt_std, \
    charge_doub_40ns_std, charge_doub_80ns_std, fwhm_doub_no_delay_std, fwhm_doub_05rt_std, fwhm_doub_1rt_std, \
    fwhm_doub_15rt_std, fwhm_doub_2rt_std, fwhm_doub_25rt_std, fwhm_doub_3rt_std, fwhm_doub_35rt_std, \
    fwhm_doub_4rt_std, fwhm_doub_45rt_std, fwhm_doub_5rt_std, fwhm_doub_55rt_std, fwhm_doub_6rt_std, \
    fwhm_doub_40ns_std, fwhm_doub_80ns_std \
        = hist_data(amp_sing, charge_sing, fwhm_sing, amp_doub_no_delay, amp_doub_05rt, amp_doub_1rt, amp_doub_15rt,
                    amp_doub_2rt, amp_doub_25rt, amp_doub_3rt, amp_doub_35rt, amp_doub_4rt, amp_doub_45rt, amp_doub_5rt,
                    amp_doub_55rt, amp_doub_6rt, amp_doub_40ns, amp_doub_80ns, charge_doub_no_delay, charge_doub_05rt,
                    charge_doub_1rt, charge_doub_15rt, charge_doub_2rt, charge_doub_25rt, charge_doub_3rt,
                    charge_doub_35rt, charge_doub_4rt, charge_doub_45rt, charge_doub_5rt, charge_doub_55rt,
                    charge_doub_6rt, charge_doub_40ns, charge_doub_80ns, fwhm_doub_no_delay, fwhm_doub_05rt,
                    fwhm_doub_1rt, fwhm_doub_15rt, fwhm_doub_2rt, fwhm_doub_25rt, fwhm_doub_3rt, fwhm_doub_35rt,
                    fwhm_doub_4rt, fwhm_doub_45rt, fwhm_doub_5rt, fwhm_doub_55rt, fwhm_doub_6rt, fwhm_doub_40ns,
                    fwhm_doub_80ns, fsps_new)

    make_plots(amp_sing_mean, charge_sing_mean, fwhm_sing_mean, amp_doub_no_delay_mean, charge_doub_no_delay_mean,
               fwhm_doub_no_delay_mean, fwhm_doub_05rt_mean, fwhm_doub_1rt_mean, fwhm_doub_15rt_mean,
               fwhm_doub_2rt_mean, fwhm_doub_25rt_mean, fwhm_doub_3rt_mean, fwhm_doub_35rt_mean, fwhm_doub_4rt_mean,
               fwhm_doub_45rt_mean, fwhm_doub_5rt_mean, fwhm_doub_55rt_mean, fwhm_doub_6rt_mean, fwhm_doub_40ns_mean,
               fwhm_doub_80ns_mean, amp_sing_std, charge_sing_std, fwhm_sing_std, amp_doub_no_delay_std,
               charge_doub_no_delay_std, fwhm_doub_no_delay_std, fwhm_doub_05rt_std, fwhm_doub_1rt_std,
               fwhm_doub_15rt_std, fwhm_doub_2rt_std, fwhm_doub_25rt_std, fwhm_doub_3rt_std, fwhm_doub_35rt_std,
               fwhm_doub_4rt_std, fwhm_doub_45rt_std, fwhm_doub_5rt_std, fwhm_doub_55rt_std, fwhm_doub_6rt_std,
               fwhm_doub_40ns_std, fwhm_doub_80ns_std, shaping, fsps_new, dest_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog="p3_double_studies2", description="Analyzing double spe data")
    parser.add_argument("--date", type=int, help='date of data acquisition (default=20190513)', default=20190513)
    parser.add_argument("--fil_band", type=str, help='folder name for data (default=full_bdw_no_nf)',
                        default='full_bdw_no_nf')
    parser.add_argument("--fsps_new", type=float, help='new samples per second (Hz) (default=500000000.)',
                        default=500000000.)
    parser.add_argument("--shaping", type=str, help='shaping amount (default=rt_1)', default='rt_1')
    args = parser.parse_args()

    double_spe_studies_2(args.date, args.fil_band, args.fsps_new, args.shaping)
