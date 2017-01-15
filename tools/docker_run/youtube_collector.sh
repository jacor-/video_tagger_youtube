..............
-- Crete docker to download the youtube dataset
..............
nvidia-docker run --name tagnet_yt -itd -v `pwd`/taggerdata:/data -v `pwd`/taggernet:/taggernet docker-registry.int.midasplayer.com/dstech/kerasstack:gpu bash
docker attach tagnet_yt;
..............
pip install --upgrade pip
pip install --upgrade youtube_dl
apt-get update
apt-get install libav-tools
pip install nltk
pip install sklearn
pip install --upgrade numpy
cd taggernet
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python datasets/dataset_creators/Youtube.py
..............
docker stop youtube_dl; docker rm youtube_dl;

