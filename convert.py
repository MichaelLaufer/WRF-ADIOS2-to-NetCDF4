import os, sys
import argparse
from collections import defaultdict

import numpy as np
from netCDF4 import Dataset
import adios2

try:
    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_size() > 1:
        parallel = True
    else:
        parallel = False

except ImportError:
    parallel = False


def progress(count, total, status=""):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = "=" * filled_len + "-" * (bar_len - filled_len)
    sys.stdout.write("\033[K")
    sys.stdout.write("[{0}] {1}% {2}\r".format(bar, percents, status))
    sys.stdout.flush()


def Locate(rank, nproc, datasize):
    extra = 0
    if rank == nproc - 1:
        extra = datasize % nproc
    num = datasize // nproc
    start = num * rank
    size = num + extra
    return start, size


def open_files(input_file, output_file, parallel=False, diskless=False):
    if parallel:
        adios2f = adios2.open(input_file, "r", comm=MPI.COMM_WORLD)
    else:
        adios2f = adios2.open(input_file, "r")
    netcdff = Dataset(
        output_file,
        "w",
        format="NETCDF4",
        parallel=parallel,
        diskless=diskless,
    )
    netcdff.set_fill_off()
    return (adios2f, netcdff)


def r_attrs(adios2f):
    attrs = adios2f.available_attributes()
    var_attrs = defaultdict(dict)  # init 2d dict
    global_attrs = {}
    for attr in attrs.keys():
        # "Dims" attribute not needed in NetCDF file
        if "/Dims" in attr:
            continue
        try:
            val = adios2f.read_attribute(attr)
        except ValueError:
            val = adios2f.read_attribute_string(attr)
        if "/" in attr:
            var, var_attrib = attr.split("/")
            var_attrs[var][var_attrib] = val
        else:
            if not attr.startswith("_DIM_"):
                global_attrs[attr] = val
    return (attrs, var_attrs, global_attrs)


def r_metadata(adios2f, attrs):
    vars = adios2f.available_variables()
    var_names = list(vars.keys())
    num_steps = adios2f.steps()
    dim_lens = {
        key[5:]: int(value["Value"])
        for key, value in attrs.items()
        if key.startswith("_DIM_")
    }

    var_dims = {}
    for var in vars:
        dims = adios2f.read_attribute_string("Dims", var)
        dims.reverse()
        var_dims[var] = dims
    typemap = {"float": "f", "int32_t": "i", "string": "c"}
    var_types = {}
    for var in vars:
        vtype = vars[var]["Type"]
        var_types[var] = typemap[vtype]
    return (var_names, num_steps, dim_lens, var_dims, var_types, vars)


def create_nc_dims(netcdff, num_steps, dim_lens):
    netcdff.createDimension("Time", size=num_steps)
    for dim in dim_lens:
        netcdff.createDimension(dim, size=dim_lens[dim])


def create_nc_vars(netcdff, var_names, var_types, var_dims):
    for var in var_names:
        netcdff.createVariable(var, var_types[var], var_dims[var])


def decomp(var, vars):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    dshape = vars[var]["Shape"].split(",")
    dshape = list(map(int, dshape))
    max_ind = np.argmax(np.array(dshape))

    # do not decompose small arrays
    if dshape[max_ind] < 50:
        start_arr = np.zeros_like(dshape)
        count_arr = np.array(dshape)
    else:
        start, count = Locate(rank, size, dshape[max_ind])
        start_arr = np.zeros_like(dshape)
        start_arr[max_ind] = start
        count_arr = np.array(dshape)
        count_arr[max_ind] = count
    return start_arr, count_arr


def r_w_data_serial(adios2f, netcdff, var_names, num_steps, vars):
    for i, var in enumerate(var_names):
        progress(i, len(var_names), status=var)
        for step in range(num_steps):
            if vars[var]["Type"] == "string":
                data = adios2f.read_string(var)
                netcdff.variables[var][step, :] = data
            else:
                data = adios2f.read(var)
                if data.ndim == 1 and len(data) == 1:
                    netcdff.variables[var][step] = data
                else:
                    netcdff.variables[var][step, :] = data


def r_w_data_parallel(adios2f, netcdff, var_names, num_steps, vars):
    comm = MPI.COMM_WORLD
    for i, var in enumerate(var_names):
        progress(i, len(var_names), status=var)
        if vars[var]["Type"] != "string":
            start_arr, count_arr = decomp(var, vars)
        for step in range(num_steps):
            if vars[var]["Type"] == "string":
                data = adios2f.read_string(var)
                netcdff.variables[var][step, :] = data
            else:
                data = adios2f.read(var, start=start_arr, count=count_arr)
                if data.ndim == 1 and len(data) == 1:
                    netcdff.variables[var][step] = data
                else:
                    netcdff.variables[var].set_collective(True)
                    if len(start_arr) == 3:
                        netcdff.variables[var][
                            step,
                            start_arr[0] : start_arr[0] + count_arr[0],
                            start_arr[1] : start_arr[1] + count_arr[1],
                            start_arr[2] : start_arr[2] + count_arr[2],
                        ] = data
                    elif len(start_arr) == 2:
                        netcdff.variables[var][
                            step,
                            start_arr[0] : start_arr[0] + count_arr[0],
                            start_arr[1] : start_arr[1] + count_arr[1],
                        ] = data
                    elif len(start_arr) == 1:
                        netcdff.variables[var][
                            step, start_arr[0] : start_arr[0] + count_arr[0]
                        ] = data


def w_global_attrs(netcdff, global_attrs):
    netcdff.setncatts(global_attrs)


def w_var_attrs(netcdff, var_attrs):
    for var in var_attrs.keys():
        for attr in var_attrs[var].keys():
            netcdff.variables[var].setncattr(attr, var_attrs[var][attr])


def close_files(adios2f, netcdff, diskless=False):
    if diskless == False:
        netcdff.close()
    adios2f.close()


def convert(input_file, output_file, parallel=False, diskless=False):
    adios2f, netcdff = open_files(
        input_file, output_file, parallel=parallel, diskless=diskless
    )
    attrs, var_attrs, global_attrs = r_attrs(adios2f)
    var_names, num_steps, dim_lens, var_dims, var_types, vars = r_metadata(
        adios2f, attrs
    )
    create_nc_dims(netcdff, num_steps, dim_lens)
    create_nc_vars(netcdff, var_names, var_types, var_dims)
    if parallel:
        r_w_data_parallel(adios2f, netcdff, var_names, num_steps, vars)
    else:
        r_w_data_serial(adios2f, netcdff, var_names, num_steps, vars)
    w_global_attrs(netcdff, global_attrs)
    w_var_attrs(netcdff, var_attrs)
    close_files(adios2f, netcdff, diskless)

    if diskless == True:
        return netcdff


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input: ADIOS2 file")
    parser.add_argument("--output", help="output: NetCDF file")
    args = parser.parse_args()
    convert(args.input, args.output, parallel=parallel)
