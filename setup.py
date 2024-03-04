from setuptools import setup

setup(
    name='decorrelategrids',
    version='0.1.0',
    description=('Given two 2D arrays, perform a windowed regression between the two, '
                 + 'aimed at isolating the correlated and the non-correlated components'),
    url='https://github.com/apasto/decorrelate-grids',
    author='Alberto Pastorutti, Marco Bartola',
    author_email='alberto.pastorutti@gmail.com',
    license='Apache-2.0',
    packages=['decorrelategrids'],
    package_data={
        'decorrelategrids': ['TODO']},
    install_requires=['numpy',
                      'xarray',
                      'scipy',
                      'netcdf4',
                      'setuptools'
                      ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering'
    ],
)
