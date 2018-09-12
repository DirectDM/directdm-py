from setuptools import setup, find_packages

with open ('directdm/_version.py') as f:
    exec(f.read())

setup(name='directdm',
      version=__version__,
      author='Fady Bishara, Joachim Brod, Benjamin Grinstein, Jure Zupan',
      author_email='joachim.brod@uc.edu',
      url='https://directdm.github.io',
      description='A python package for dark matter direct detection',
      long_description=""" This package contains classes for Wilson coefficients
                           for dark matter -- standard model interactions, 
                           in effective theories at various scales. It allows for  
                           renormalization-group running, and matching between the 
                           different effective theories.""",
      license='MIT',
      packages=find_packages(),
      install_requires=['numpy', 'scipy', 'setuptools']
    )
