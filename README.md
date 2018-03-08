# AmoveoMinerGpuCuda
Amoveo Cryptocurrency Miner for Gpu work to be used with [AmoveoPool.com](http://AmoveoPool.com). This only works for Windows NVidia cards with Cuda.

Tested Gpu Speeds:
* GTX1080 TI: 780 Mh/s  - Suggested BlockSize: 192
* GTX1060:    357 Mh/s  - Suggested BlockSize: 256
* GTX1050:    171 Mh/s  - Suggested BlockSize: 256
* Tesla K80:  151 Mh/s  - Suggested BlockSize: 512
* 750TI:      133 Mh/s  - Suggested BlockSize: 256

Try various BlockSize setting values. Optimal setting for BlockSize is very personal to your system. Try BlockSize values like 128, 192, 256, or 512.

## Windows

### Run Dependencies
* Install Visual Studio 2015 (Community Edition is free.)
* Install Cuda 9.1

### Releases

   [Latest pre-built releases are here](https://github.com/Mandelhoff/AmoveoMinerGpuCuda/releases)


### Run
   
Example Usage:  
```
AmoveoMinerGpuCuda.exe BPA3r0XDT1V8W4sB14YKyuu/PgC6ujjYooVVzq1q1s5b6CAKeu9oLfmxlplcPd+34kfZ1qx+Dwe3EeoPu0SpzcI=
```

Advanced Usage Template:
```
AmoveoMinerGpuCuda.exe <Base64AmoveoAddress> <CudaDeviceId> <BlockSize> <NumBlocks> <RandomSeed> <PoolUrl>
```
* CudaDeviceId is optional an defaults to 0.
* BlockSize is optional and defaults to 256.
* NumBlocks is optional and defaults to 65536
* RandomSeed is optional. Set this if you want multiple miners using the same address to avoid nonce collisions.
* PoolUrl is optional and defaults to http://amoveopool.com/work


### Build
The Windows releases are built with Visual Studio 2015 with Cuda, RestCPP, boost, and openSSL.

