from setuptools import setup
from setuptools import find_packages


setup(
    name='tfcf',
    packages=find_packages(),

    version='0.0.0',

    license='MIT',

    description='A tensorflow-based recommender system.',

    author='Tzu-Wei Sung',
    author_email='windqaq@gmail.com',

    url='https://github.com/WindQAQ/tf-recsys',

    keywords=['recommender system', 'collaborative filtering',
              'tensorflow', 'SVD', 'SVD++'],

    install_requires=[
        'requests',
        'pandas',
        'numpy',
        'tensorflow>=1.2.0',
    ],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],

    python_requires='>=3',
)
