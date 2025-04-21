# Installs necessary environment and dependencies on a clean machine.
# Most useful for VMs.

"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
source ~/.bashrc

git clone https://github.com/joel-shor/nucleobench.git

micromamba env create -f environment.yml
