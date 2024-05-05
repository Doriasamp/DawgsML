from setuptools import setup, find_packages


def read_readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()
    return long_description


setup(
    name='DawgsML',
    version='0.0.3',
    author='Andrea Marchi',
    author_email='andreaphilly@hotmail.com',
    description='A simple library for machine learning without a requirements.txt',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/Doriasamp/DawgsML.git',
    packages=find_packages(),
    install_requires=[],  # nothing here
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
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
