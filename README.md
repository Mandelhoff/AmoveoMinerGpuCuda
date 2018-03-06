# AmoveoMinerGpuCuda
Amoveo Cryptocurrency Miner for Gpu work to be used with [AmoveoPool.com](http://AmoveoPool.com). This only works for Windows NVidia cards with Cuda.

Tested Gpu Speeds:
* GTX1060: 240 Mh/s
* GTX1050: 171 Mh/s
* Tesla K80: 151 Mh/s
* 750TI: 101 Mh/s


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
AmoveoMinerGpuCuda.exe <Base64AmoveoAddress> <CudaDeviceId> <BlockSize> <NumBlocks> <PoolUrl>
```
* CudaDeviceId is optional an defaults to 0.
* BlockSize is optional and defaults to 192.
* NumBlocks is optional and defaults to 65536
* PoolUrl is optional and defaults to http://amoveopool.com/work


### Build
The Windows releases are built with Visual Studio 2015 with Cuda, RestCPP, boost, and openSSL.

