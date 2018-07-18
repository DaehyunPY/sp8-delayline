from distutils.core import setup


setup(
    name='sp8-delayline',
    version='20180714',
    author='Daehyun You',
    author_email='daehyun.park.you@gmail.com',
    description='',
    long_description=open('README.md').read(),
    license='MIT',
    packages=['dltools'],
    install_requires=['numpy', 'scipy', 'numba', 'pyspark'],
)
