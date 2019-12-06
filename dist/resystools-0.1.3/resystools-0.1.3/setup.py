
from distutils.core import setup
setup(
  name = 'resystools',         # How you named your package folder (MyLib)
  packages = ['resystools'],   # Chose the same as "name"
  version = '0.1.3',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'a lib with some recommendation algorithms',   # Give a short description about your library
  author = 'Trong Duc Le',                   # Type in your name
  author_email = 'trongduclebk@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/DucLeTrong/resystools',   # Provide either the link to your github or to your website
  download_url = '',    # I explain this later on
  keywords = ['RECOMMENDATION', 'RECOMMENDER'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'sklearn',
          'pandas',
          'numpy'
      ],
  classifiers=[
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
# if __name__ == '__main__':
#     setup(**setup_args, install_requires=install_requires)