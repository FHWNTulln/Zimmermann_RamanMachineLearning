import re
import struct
import logging

import numpy as np

opus_conv_logger = logging.getLogger(__name__)


def convert_opus(file, meta=False):
    """Extract data from a binary OPUS file.

    Args:
        file (str): Path to OPUS file.
        meta (bool, optional): !TBA! Whether metadata of the file should be returned. Defaults to False.

    Returns:
        array: Spectral data from the file, with wavenumbers in the first and intensities in the second column.
    """
    opus_conv_logger.debug(f"Loading OPUS binary file {file}")
    with open(file, "rb") as f:
        data = f.read()

    ramanDataRegex = re.compile(br"""
        \x00{5}NPT\x00{3}\x02\x00(.{4})                 # Number of points
        FXV\x00\x01\x00\x04\x00(.{8})                   # First wavenumber
        LXV\x00\x01\x00\x04\x00(.{8})                   # Last wavenumber
        CSF\x00.{12}MXY\x00.{12}                        # Not used
        MNY\x00.{12}DPF\x00.{8}                         # Not used
        DAT\x00.{4}(.{10})\x00\x00                      # Date
        TIM\x00.{4}(.{16}).{4}                          # Time
        DXU\x00.{8}                                     # Not used
        END\x00{9}(.*?)\x00{4}NPT                      # Raman Data
    """, re.VERBOSE | re.DOTALL)

    opus_conv_logger.debug("Extracting data...")
    mo = ramanDataRegex.search(data)
    npt, fxv, lxv, dat, tim, ints = mo.groups()

    npt = struct.unpack("<I", npt)[0]
    fxv = struct.unpack("<d", fxv)[0]
    lxv = struct.unpack("<d", lxv)[0]
    dat = dat.decode("ascii")
    tim = tim.decode("ascii")

    wns = np.linspace(fxv, lxv, npt)
    ints = np.asarray(struct.unpack("<" + "f"*npt, ints))

    data_out = np.column_stack((wns, ints))

    # TODO: Handle metadata
    metadat = []

    if meta:
        return data_out, metadat
    else:
        return data_out
