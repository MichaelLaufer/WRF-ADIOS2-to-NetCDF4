# WRF-ADIOS2-to-NetCDF4
This tool converts WRF outputted ADIOS2 files to NetCDF4 files for backwards compatibility.

Distributed memory (MPI) support is available for file conversions, but was not found to offer performance improvements over the serial version, but it can be used for large dataset that do not fit in the memory of a single node.

For batch jobs that need to convert multiple files, Gnu Parallel should be used to convert multiple files concurrently, which should reduce the conversion time significantly.
## Usage
There are 2 ways to use this script: File conversion, and In-line implicit conversion ("diskless).

### File conversion:
This will convert the ADIOS2 file and save it as a file in the output location.<br/>
*Serial functionality (recommended)*
```
python convert.py --input wrfout_d01_2018-06-17_02\:00\:00/ --output ./converted/converted.nc
```
*MPI functionality*
```
mpirun -np 2 -host host:1,host2:1 convert.py --input wrfout_d01_2018-06-17_02\:00\:00/ --output ./converted/converted.nc
```
### In-line implicit conversion ("diskless"):<br/>
This allows the conversion function to be used in an existing netcdf4-python processing, and returns a NetCDF Dataset object while keeping the dataset in memory (not on disk).
```
import convert
netcdf4_Dataset = convert.convert(input_file, output_file, diskless=True)
...
...
```

## Dependencies
netcdf4-python<br/>
ADIOS2 (with Python bindings, Blosc compressor)<br/>
For MPI functionality, use parallel version of both
