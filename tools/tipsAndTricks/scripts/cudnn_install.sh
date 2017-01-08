echo -e "\nexport LD_LIBRARY_PATH=/home/jose/Downloads/cuda:$LD_LIBRARY_PATH" >> ~/.bashrc

sudo cp /home/jose/Downloads/cuda/include/cudnn.h /usr/local/cuda-7.0/include
sudo cp /home/jose/Downloads/cuda/lib64/* /usr/local/cuda-7.0/lib64


export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib64:$LD_LIBRARY_PATH
export CPATH=/usr/local/cuda-7.0/include:$CPATH
export LIBRARY_PATH=/usr/local/cuda-7.0/lib64:$LD_LIBRARY_PATH

