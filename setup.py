from distutils.core import setup


setup(
    name='sp8-delayline',
    version='20180602',
    author='Daehyun You',
    author_email='daehyun.park.you@gmail.com',
    description='',
    long_description=open('README.md').read(),
    license='MIT',
    packages=['sp8tools'],
    install_requires=['numpy', 'scipy', 'numba', 'pyspark', 'distributed'],
)
