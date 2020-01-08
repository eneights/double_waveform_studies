from p3_functions import *

# Amp, charge, FWHM cutoffs for each sample rate and shaping
# Creates arrays of approximate amp, charge, FWHM cutoffs for each sample rate and shaping
# if amp > cutoff amp or charge > cutoff charge or FWHM > cutoff FWHM, waveform is MPE
# else, waveform is SPE
# Do above for all SPE and MPE waveforms at each sample rate and shaping for different combos of cutoffs
# Find % false MPEs and % false SPEs for each combo
