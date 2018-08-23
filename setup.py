from distutils.core import setup


setup(
    name='sp8-delayline',
    version='20180823',
    author='Daehyun You',
    author_email='daehyun.park.you@gmail.com',
    description='',
    long_description=open('README.md').read(),
    license='MIT',
    packages=['dltools'],
    py_modules=['sp8export'],
    install_requires=['pyyaml', 'numpy', 'scipy', 'numba', 'pyspark'],
)
