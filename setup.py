from setuptools import setup, find_packages

setup(
    name='analyze_team_6',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='Analyze team_6 python package',
    long_description=open('README.md').read(),
    install_requires=['numpy','pandas'],
    url='https://github.com/Prescam3',
    author='Presca Mashamaite',
    author_email='prescam3@gmail.com'
)