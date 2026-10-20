from setuptools import setup, find_packages

setup(
    name='ioos_model_comparisons',
    version='0.0.2',
    packages=find_packages(),
    # YAML configuration (regions, models, comparisons, RTOFS, portal, site)
    package_data={'ioos_model_comparisons': ['config/*.yml', 'config/regions/*.yml']},
    include_package_data=True,
    url='github.com/rucool/ioos_model_comparisons/',
    license='',
    author='mikesmith',
    author_email='michaesm@marine.rutgers.edu',
    description='Rutgers Hurricane Model Comparisons'
)
