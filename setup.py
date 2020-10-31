from setuptools import setup, find_packages

setup(name='pure_ml',
    version='0.1.0',
    description='ML with pure Python',
    author='Ollie Sellers',
    author_email='olliejsellers@gmail.com',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas'
        ],
    zip_safe=True)
