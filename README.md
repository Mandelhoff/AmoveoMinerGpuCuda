# AmoveoMinerGpuCuda
Amoveo Cryptocurrency Miner for Gpu work to be used with [AmoveoPool.com](http://AmoveoPool.com). This only works for Windows NVidia cards with Cuda.

Tested Gpu Speeds:
* Tesla P100: 1920 Mh/s  - Suggested BlockSize: 192, Numblocks: 168
* GTX1080 TI: 1500 Mh/s  - Suggested BlockSize: ?
* GTX1060 6GB: 901 Mh/s  - Suggested BlockSize: 64
* GTX1050:    430 Mh/s  - Suggested BlockSize: 64
* Tesla K80:  301 Mh/s  - Suggested BlockSize: 128
* 750TI:      238 Mh/s  - Suggested BlockSize: 32

Default BlockSize is 64.
Default NumBlocks is 96.
Default SuffixMax is 65536.

* Try various BlockSize setting values. Optimal setting for BlockSize is very personal to your system. Try BlockSize values like 96, 64, 32, or 128. A higher BlockSize is almost always better, but too high will crash the miner.
* If you get too much OS lag, reduce the SuffixMax setting (at the cost of a some hash rate).
* If your Memory Controller Load is constantly at 100%, you may want to try lowering your NumBlocks.

Best Settings from My Tests:
* Gtx1060: BlockSize=64, NumBlocks=96
* Gtx1050: BlockSize=64, NumBlocks=90
* Tesla K80: BlockSize=128, NumBlocks=128
* 750Ti: BlockSize=32, NumBlocks=64



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
AmoveoMinerGpuCuda.exe <Base64AmoveoAddress> <CudaDeviceId> <BlockSize> <NumBlocks> <RandomSeed> <SuffixMax> <PoolUrl>
```
* CudaDeviceId is optional an defaults to 0.
* BlockSize is optional and defaults to 256.
* NumBlocks is optional and defaults to 65536
* RandomSeed is optional. Set this if you want multiple miners using the same address to avoid nonce collisions.
* SuffixMax optional and defaults to 65536. Do NOT use anything higher than 65536. Lower numbers reduce OS lag and will reduce hash rate by a few percent.
* PoolUrl is optional and defaults to http://amoveopool.com/work


### Build
The Windows releases are built with Visual Studio 2015 with Cuda, RestCPP, boost, and openSSL.



Donations are welcome:
* Amoveo: BPA3r0XDT1V8W4sB14YKyuu/PgC6ujjYooVVzq1q1s5b6CAKeu9oLfmxlplcPd+34kfZ1qx+Dwe3EeoPu0SpzcI=
* Ethereum: 0x74e0aF0522024f2dd94F0fb9B82d13782ECCaaF5
