/*************************************************************************************************/
                                          CLONE NCNN
/*************************************************************************************************/
Step 1 
git clone https://github.com/Tencent/ncnn.git
Step 2 
cd ncnn
Step 3 
git submodule update --init

/*************************************************************************************************/
                              DOWNLOAD VULKAN SDK AND SET ENV
/*************************************************************************************************/
Step 1
wget https://sdk.lunarg.com/sdk/download/1.2.154.0/linux/vulkansdk-linux-x86_64-1.2.154.0.tar.gz?Human=true -O vulkansdk-linux-x86_64-1.2.154.0.tar.gz
Step 2
tar -xf vulkansdk-linux-x86_64-1.2.154.0.tar.gz
Step 3
export VULKAN_SDK=$(pwd)/1.2.154.0/x86_64
Step 4
sudo apt-get update -y
Step 5
sudo apt-get install -y libvulkan-dev


/*************************************************************************************************/
                                          RUN WITH GPU
/*************************************************************************************************/
Step 1 
cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=ON -DNCNN_SYSTEM_GLSLANG=ON -DNCNN_BUILD_EXAMPLES=ON -DDLIB_USE_CUDA=1 -DCMAKE_TOOLCHAIN_FILE=../toolchains/jetson.toolchain.cmake ..
Step 2
make -j$(nproc)

/*************************************************************************************************/
                                  SHOW ALL DEVICE INFORMATION
/*************************************************************************************************/
gst-device-monitor-1.0
"v4l2src device=/dev/video0 ! image/jpeg, width=(int)1280, height=(int)720, framerate=30/1 ! jpegdec ! videoconvert ! appsink"
# Project_FaceRecognize
