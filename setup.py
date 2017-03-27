#!/usr/bin/env python
"""minimal setup script."""

from distutils.core import setup

# Package Version
from minimal import __version__ as version

setup(
    name='minimal',
    version=version,

    description=('MatrIx regularizatioN In MAchine Learning'),
    long_description=open('README.md').read(),
    author='Samuele Fiorini, Annalisa Barla',
    author_email='samuele.fiorini@dibris.unige.it, annalisa.barla@unige.it',
    maintainer='Samuele Fiorini',
    maintainer_email='samuele.fiorini@dibris.unige.it',
    url='https://github.com/samuelefiorini/minimal',
    # download_url='https://github.com/slipguru/adenine/tarball/'+version,
    download_url='https://github.com/samuelefiorini/minimal/archive/master.zip',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'License :: OSI Approved :: BSD License',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS'
    ],
    license='FreeBSD',

    packages=['minimal'],
    install_requires=['numpy (>=1.10.1)',
                      'scipy (>=0.16.1)',
                      'scikit-learn (>=0.18.1)',
                      'matplotlib (>=2.0.1)',
                      'seaborn (>=0.7.0)'],
    scripts=['scripts/mini_train.py', 'scripts/mini_test.py'],
)
