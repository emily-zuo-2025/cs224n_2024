arch=$(uname -m)
if [ "$arch" = "arm64" ]; then
  installer=Miniconda3-latest-MacOSX-arm64.sh
else
  installer=Miniconda3-latest-MacOSX-x86_64.sh
fi
curl -fsSL -o ~/miniconda.sh https://repo.anaconda.com/miniconda/${installer}
bash ~/miniconda.sh -b -p $HOME/miniconda3
rm ~/miniconda.sh
$HOME/miniconda3/bin/conda init zsh

source ~/.zshrc

conda --version
which conda
conda env list