from setuptools import setup, find_packages

setup(
    name='transfer',
    version='0.0.1',
    description='Transfer learning for educational models.',
    author='Chris Brooks, Rene Kizilcec, Josh Gardner, Quan Nguyen, Renzhe Yu',
    author_email='jpgard@cs.washington.edu',
    license='MIT',
    url='https://github.com/educational-technology-collective/cross-institutional-transfer-learning-facct-2023',
    packages=find_packages(),
    install_requires=['pandas', 'numpy', 'seaborn'],
)
