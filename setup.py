def get_version():

    import os
    import sys

    sys.path.append(os.path.abspath('naiveb'))
    from version_info import VERSION as version
    sys.path.pop()

    return version

def get_readme():

    with open('README.md') as f:
        return f.read()

setup(

  name='epios',
  version=get_version(),
  description='Implementation of Naive Bayes Classifier methods.',
  long_description=get_readme(),
  #license=,
  # author='Antonio Mastromarino',
  # author_email='',
  maintainer='Antonio Mastromarino',
  maintainer_email='antonio.mastromarino@wolfson.ox.ac.uk',
  url='https://github.com/AntonioVitoMastromarino/NaiveB',

  packages=find_packages(include=('naiveb', 'naiveb.*')),
    install_requires=[
      'numpy>=1.8',
      'matplotlib',
      'pandas>=1.4',
      'scipy'
      'python_version>=3.8',
    ],
    extras_require={
      'docs': [
        # Sphinx for doc generation. Version 1.7.3 has a bug:
        'sphinx>=1.5, !=1.7.3',
        # Nice theme for docs
        'sphinx_rtd_theme',
      ],
    },
)
