from setuptools import setup, find_packages

setup(packages=find_packages(include=('empirbayes', 'empirbayes.*')),
      install_requires=['numpy', 'matplotlib', 'pandas', 'scipy'],
      name='empirbayes')
