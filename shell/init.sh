mkdir results
mkdir results/dict
mkdir results/models
mkdir results/logs

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install  dgl -f https://data.dgl.ai/wheels/cu116/repo.html
pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
pip install torch_geometric
pip install loguru numpy scipy rich tqdm matplotlib
pip install torchmetrics
pip install networkx==2.6.3
pip install ogb
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu11 dask-cudf-cu11 cuml-cu11 cugraph-cu11 cuspatial-cu11 cuproj-cu11 cuxfilter-cu11 cucim