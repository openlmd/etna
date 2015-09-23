from distutils.core import setup

setup(name="robpath",
      version="0.2.0",
      description="Robot Path Planner for Laser Metal Deposition",
      author="Jorge Rodriguez-Araujo",
      author_email="jorge.rodriguez@aimen.es",
      url="http://www.aimen.es",
      packages=["robpath"],
      data_files=[('ui', ['robpath/robpath.ui']),
                  ('icons', ['robpath/logo.jpg'])],
      install_requires=["mayavi >= 4.10.0"]
      )
