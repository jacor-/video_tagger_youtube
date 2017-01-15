..............
-- Crete docker to train the network
..............
https://JoseCordero@bitbucket.org/JoseCordero/research_tagger_network.git taggernet;
nvidia-docker run --name tag_learn -itd -v `pwd`/taggerdata:/data -v `pwd`/taggernet:/taggernet docker-registry.int.midasplayer.com/dstech/kerasstack:gpu bash;
docker attach tag_learn;
..........
pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
python taggernet/tests/combinations_synthetic_datasets/write_tests_to_run.py > taggernet/tests/combinations_synthetic_datasets/tests_to_be_executed.txt;
cd taggernet;
bash tests/combinations_synthetic_datasets/tests_to_be_executed.txt
..........
docker stop tag_learn; docker rm tag_learn;

