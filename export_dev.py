from os import chdir
from os.path import abspath, dirname
from sys import argv

from yaml import load as load_yaml

from sp8tools import with_unit, none_field, uniform_electric_field, ion_spectrometer, electron_spectrometer


if __name__ == '__main__':
    if len(argv) == 1:
        raise ValueError("Usage: program config")
    elif len(argv) == 2:
        config_filename = abspath(argv[1])
        working_directory = dirname(config_filename)
        print("Change working directory to '{}'...".format(working_directory))
        chdir(working_directory)
    else:
        raise ValueError("Too many arguments!: '{}'".format(argv[1:]))

    with open(config_filename, 'r') as f:
        print("Reading config file: '{}'...".format(config_filename))
        config = load_yaml(f)

    spectrometer = {k: with_unit(v) for k, v in config['spectrometer'].items()}
    ion_acc = (
            uniform_electric_field(length=spectrometer['length_of_GepReg'],
                                   electric_field=(
                                           (spectrometer['electric_potential_of_Ion2nd'] -
                                            spectrometer['electric_potential_of_IonMCP']) /
                                           spectrometer['length_of_GepReg'])) *
            uniform_electric_field(length=spectrometer['length_of_AccReg'],
                                         electric_field=(
                                                 (spectrometer['electric_potential_of_Ion1nd'] -
                                                  spectrometer['electric_potential_of_Ion2nd']) /
                                                 spectrometer['length_of_AccReg'])) *
            uniform_electric_field(length=spectrometer['length_of_LReg'],
                                   electric_field=(
                                           (spectrometer['electric_potential_of_Electron'] -
                                            spectrometer['electric_potential_of_Ion1nd']) /
                                           (spectrometer['length_of_LReg'] +
                                            spectrometer['length_of_DReg']))))

    ele_acc = (
            none_field(length=spectrometer['length_of_DReg']) *
            uniform_electric_field(length=spectrometer['length_of_DReg'],
                                   electric_field=(
                                           (spectrometer['electric_potential_of_Ion1nd'] -
                                            spectrometer['electric_potential_of_Electron']) /
                                           (spectrometer['length_of_LReg'] +
                                            spectrometer['length_of_DReg']))))
