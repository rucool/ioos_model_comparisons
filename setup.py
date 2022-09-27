from setuptools import setup

setup(
    name='ioos_model_comparisons',
    version='0.0.2',
    packages=['ioos_model_comparisons',
              'data', 
              'scripts',
              'scripts.gliders',
              'scripts.harvest', 
              'scripts.maps',
              'scripts.met',
              'scripts.misc', 
              'scripts.profiles',
              'scripts.transects',
              ],
    url='github.com/rucool/ioos_model_comparisons/',
    license='',
    author='mikesmith',
    author_email='michaesm@marine.rutgers.edu',
    description='Rutgers Hurricane Model Comparisons'
)
