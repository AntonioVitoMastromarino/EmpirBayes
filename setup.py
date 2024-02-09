from setuptools import setup, find_packages

def get_readme():

    with open('README.md') as f:
        return f.read()

setup(

  name='naiveb',
  description='Implementation of Naive Bayes Classifier methods.',
  long_description=get_readme(),
  version='0.0.0',
  license='MIT License',
  author='Antonio Mastromarino',
  author_email='antonio.mastromarino@wolfson.ox.ac.uk',
  maintainer='Antonio Mastromarino',
  maintainer_email='antonio.mastromarino@wolfson.ox.ac.uk',
  url='https://github.com/AntonioVitoMastromarino/NaiveB',

  packages=find_packages(include=('naiveb', 'naiveb.*')),
    install_requires=[
      'numpy',
      'matplotlib',
      'pandas',
      'scipy',
    ],
)
