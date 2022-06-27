from setuptools import setup

setup(
    name='hurricanes',
    version='0.0.2',
    packages=['hurricanes',
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
    url='github.com/rucool/hurricanes/',
    license='',
    author='mikesmith',
    author_email='michaesm@marine.rutgers.edu',
    description='Rutgers Hurricane Model Comparisons'
)
