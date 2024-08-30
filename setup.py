from setuptools import find_packages, setup


def get_requirements():

    with open('requirements.txt') as f:
        requirements = [
            line.strip()
            for line in f.readlines()
            if not line.startswith('#') and len(line.strip()) > 0]

    requirements = [str.replace(r, '==', '>=')
                    for r in requirements]

    return requirements


setup(
    name='aithena',
    version='0.1',
    url='https://github.com/Athena1994/Aithena',
    packages=find_packages(include=['aithena.*']),
    install_requires=get_requirements()
)
