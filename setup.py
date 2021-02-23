from setuptools import setup

setup(name='raskls',
      version='1.0',
      description='Various homemade estimators that I use often enough that I wanted to wrap them all up.',
      url='https://github.com/richardseifert/raskls',
      author='Richard Seifert',
      author_email='seifertricharda@gmail.com',
      license='MIT',
      packages=['raskls'],
      install_requires=[
            'sklearn',
            'numpy',
            'pandas',
      ],
      zip_safe=False,
      include_package_data=True)
