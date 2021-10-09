import setuptools

setuptools.setup(
    name='epicas',
    version='0.1.0',
    author='Duy Cao',
    author_email='caominhduy@gmail.com',
    description='An AutoML framework designed for epidemiological forecasting',
    long_description='''
Epicas stands for Epidemiological Forecasting. Epicas is an AutoML framework based on TensorFlow and statsmodels.

For instructions or details about source code, please see at https://github.com/caominhduy/epicas

This package is still under development. Feel free to contact developer at caominhduy@gmail.com if any issue arises.
    ''',
    url='https://github.com/caominhduy/epicas',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ],
    packages=setuptools.find_packages(exclude=("tests",)),
    include_package_data=True,
    install_requires=[
        'tensorflow>=2.6.0',
        'keras-self-attention>=0.50.0',
        'pandas>=1.2.4',
        'statsmodels>=0.13.0',
        'tqdm'
    ],
    keywords=[
        'AutoML',
        'ARIMA',
        'attention',
        'epidemiology',
        'forecast',
        'infectious',
        'machine learning',
        'TensorFlow'
        ],
    python_requires='>=3.7.1'
)
