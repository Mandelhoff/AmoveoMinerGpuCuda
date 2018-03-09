# AmoveoMinerGpuCuda
Amoveo Cryptocurrency Miner for Gpu work to be used with [AmoveoPool.com](http://AmoveoPool.com). This only works for Windows NVidia cards with Cuda.

Tested Gpu Speeds:
* GTX1080 TI: 870 Mh/s  - Suggested BlockSize: ?
* GTX1060:    458 Mh/s  - Suggested BlockSize: 64
* GTX1050:    271 Mh/s  - Suggested BlockSize: 64
* Tesla K80:  250 Mh/s  - Suggested BlockSize: 128
* 750TI:      177 Mh/s  - Suggested BlockSize: 32

Default BlockSize is now 64.
Default NumBlocks is now 96.

Best for me:
* Gtx1060: BlockSize=64, NumBlocks=96
* Gtx1050: BlockSize=64, NumBlocks=64
* Tesla K80: BlockSize=128, NumBlocks=128
* 750Ti: BlockSize=32, NumBlocks=64

Try various BlockSize setting values. Optimal setting for BlockSize is very personal to your system. Try BlockSize values like 96, 64, 32, or 128. A higher BlockSize is almost always better, but too high will crash the miner.

If your Memory Controller Load is constantly at 100%, you may want to try lowering your NumBlocks.



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



Donations are welcome:
* Amoveo - BPA3r0XDT1V8W4sB14YKyuu/PgC6ujjYooVVzq1q1s5b6CAKeu9oLfmxlplcPd+34kfZ1qx+Dwe3EeoPu0SpzcI=
* Ethereum - 0x74e0aF0522024f2dd94F0fb9B82d13782ECCaaF5
