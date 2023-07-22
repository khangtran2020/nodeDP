mkdir results
mkdir results/dict
mkdir results/models
mkdir results/logs

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -c nvidia -y
conda install -c dglteam/label/cu113 dgl -y
conda install pyg -c pyg -y
conda install pandas numpy scipy tqdm matplotlib seaborn -y