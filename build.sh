cp main.cpp main.cu
nvcc -O3 -c base64.cpp
nvcc -O3 -c -std=c++11 poolApi.cpp
nvcc -O3 -std=c++11 -Wno-deprecated-gpu-targets -c main.cu
nvcc -O3 -o AmoveoMinerGpuCuda -Wno-deprecated-gpu-targets\
    -std=c++11\
    main.o base64.o poolApi.o\
    -lstdc++ -lcpprest -lboost_system -lcrypto -lssl -lpthread -lcuda -lcudart
rm *.o main.cu
