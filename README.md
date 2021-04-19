/============/ \
CLONE DLIB \
/===========/ \
git clone https://github.com/davisking/dlib \
\
/============/ \
CLONE NCNN \
/===========/ \
Step 1  \
git clone https://github.com/Tencent/ncnn.git \
Step 2  \
cd ncnn \
Step 3  \
git submodule update --init \
\
/============/ \
DOWNLOAD VULKAN SDK AND SET ENV \
/============/ \
Step 1 \
wget https://sdk.lunarg.com/sdk/download/1.2.154.0/linux/vulkansdk-linux-x86_64-1.2.154.0.tar.gz?Human=true -O \
vulkansdk-linux-x86_64-1.2.154.0.tar.gz \
Step 2 \
tar -xf vulkansdk-linux-x86_64-1.2.154.0.tar.gz \
Step 3 \
export VULKAN_SDK=$(pwd)/1.2.154.0/x86_64 \
Step 4 \
sudo apt-get update -y \
Step 5 \
sudo apt-get install -y libvulkan-dev \
Step 6
sudo apt-get install libopenblas-dev liblapack-dev
sudo apt-get install protobuf-compiler libprotobuf-dev
apt-get install libatlas-base-dev 
sudo apt install libcanberra-gtk-module libcanberra-gtk3-module
 \
/============/ \
RUN WITH GPU \
/============/ \
Step 1 
cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=ON -DGLSLANG_TARGET_DIR=ON \
-DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1 .. \
Step 2 \
make -j$(nproc) \
 \
/============/ \
SHOW ALL DEVICE INFORMATION \
/============/ \
gst-device-monitor-1.0 \
"v4l2src device=/dev/video0 ! image/jpeg, width=(int)1280, height=(int)720, \
framerate=30/1 ! jpegdec ! videoconvert ! appsink" \

kyo@kyo:~$ export PATH=${PATH}:/usr/local/cuda/bin
kyo@kyo:~$ export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
kyo@kyo:~$ sudo ln -s /usr/local/cuda-8.0 /usr/local/cuda
[sudo] password for kyo: 
kyo@kyo:~$ 
kyo@kyo:~$ sudo ln -s /usr/local/cuda-10.2.89/usr/local/cuda
kyo@kyo:~$ export CPATH=/usr/local/cuda-10.2.89/targets/x86_64-linux/include:$CPATH
kyo@kyo:~$ export LD_LIBRARY_PATH=/usr/local/cuda-10.2.89/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
kyo@kyo:~$ export PATH=/usr/local/cuda-10.2.89/bin:$PATH
kyo@kyo:~$ CUDA_INC_PATH=/usr/local/cuda/include
kyo@kyo:~$ CUDA_LIB_PATH=/usr/local/cuda/lib
kyo@kyo:~$ env | grep CUDA

cmake -DCMAKE_CUDA_COMPILER:PATH=/usr/local/cuda/bin/nvcc

https://kezunlin.me/post/4eb7fcec/
# Project_FaceRecognize \

sudo systemctl restart nvargus-daemon

