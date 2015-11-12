import os
from setuptools import setup, find_packages


CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Topic :: Scientific/Engineering',
]


setup(name='properscoring',
      description='Proper scoring rules in Python',
      long_description=(open('README.rst').read()
                        if os.path.exists('README.rst')
                        else ''),
      version='0.1',
      license='Apache',
      classifiers=CLASSIFIERS,
      author='The Climate Corporation',
      author_email='eng@climate.com',
      url='https://github.com/TheClimateCorporation/properscoring',
      install_requires=['numpy', 'scipy'],
      tests_require=['nose'],
      packages=find_packages())
