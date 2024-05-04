from setuptools import setup, find_packages

setup(
    name='DawgsML',
    version='0.0.1',
    author='Andrea Marchi',
    author_email='andreaphilly@hotmail.com',
    description='A simple library for machine learning without a requirements.txt',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Doriasamp/DawgsML.git',
    packages=find_packages(),
    install_requires=[],  # nothing here
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Data Scientists',
        'Intended Audience :: Data Analysts',
        'Intended Audience :: Machine Learning Engineers',
        'Topic :: Software Development :: Libraries',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    python_requires='>=3.9',
    keywords='machine learning, data processing, custom data structures, pandas-like, data analysis, ml, '
             'data science, statistics, numpy, deep learning, preprocessing, pandas simulation',
    project_urls={
        'Source': 'https://github.com/Doriasamp/DawgsML.git',
    },
)
