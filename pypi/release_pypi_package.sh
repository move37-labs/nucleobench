# Simple workflow to build and deploy a new PyPi package.
#
# To run:
# bash pypi/release_pypi_package.sh
#

# Clear old version.
rm dist/*

# Pick which version to build.
# cp pypi/pyproject.full.toml pyproject.toml
cp pypi/pyproject.optimizers.toml pyproject.toml

# Build the package.
python -m build

# Upload to test PyPi index.
python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# Upload to real PyPi index.
#python -m twine upload dist/*
