from p3_functions import *


# Plots a waveform for the user to view (does not save)
def plot_waveform(start, end, fil_band, folder, d_folder):
    for i in range(start, end + 1):
        show_waveform(Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/waveforms/' + str(fil_band) +
                           '/d3/' + str(folder) + '/' + str(d_folder) + r'/D3--waveforms--%05d.txt' % i), 'd3')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog="plot_waveform", description="Plotting waveforms")
    parser.add_argument("--start", type=int, help='first file number to plot', default=0)
    parser.add_argument("--end", type=int, help='last file number to plot', default=0)
    parser.add_argument("--fil_band", type=str, help='folder name for data', default='full_bdw_no_nf')
    parser.add_argument("--folder", type=str, help='shaping folder', default=' ')
    parser.add_argument("--d_folder", type=str, help='downsampled/digitized folder', default=' ')
    args = parser.parse_args()

    plot_waveform(args.start, args.end, args.fil_band, args.folder, args.d_folder)