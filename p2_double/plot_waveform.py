from p2_functions import *


# Plots a waveform for the user to view (does not save)
def plot_waveform(fil_band, folder, delay_folder):

    for filename in os.listdir(Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/waveforms/' +
                                    str(fil_band) + '/d2/' + str(folder) + '/' + str(delay_folder))):
        file = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/waveforms/' +
                                    str(fil_band) + '/d2/' + str(folder) + '/' + str(delay_folder)) / filename
        show_waveform(file, 'd2')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog="plot_waveform", description="Plotting waveforms")
    parser.add_argument("--fil_band", type=str, help='folder name for data', default='full_bdw_no_nf')
    parser.add_argument("--folder", type=str, help='folder within d2', default=' ')
    parser.add_argument("--delay_folder", type=str, help='delay folder name', default='no_delay')
    args = parser.parse_args()

    plot_waveform(args.fil_band, args.folder, args.delay_folder)