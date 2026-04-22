from setuptools import setup

setup(
    name='drs4_parser',
    version='0.1',
    packages=['parser', 'pals', 'events_filter', 'time_analysis', 'energy_analysis'],
    package_dir={'': 'lifetime_parser'},
    url='',
    license='',
    author='Achiya Yosef Amrusi',
    author_email='ahia.amrosi@mail.huji.ac.il',
    description='parser for pals measurments using drs4 from binary files to pals histograms'
)
