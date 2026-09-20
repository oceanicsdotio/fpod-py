"""
Read and analyse FPOD FP1/FP3 file format.

This implementation is based on the original FPOD FP1 file format specification from Chelonia, and the R/C++ language implementation of the FPOD FP1 reader.
"""
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, cast
from enum import IntEnum
from pandas import DataFrame, Timestamp, to_timedelta, set_option, Series
from matplotlib.pyplot import subplots
from click import echo, group, argument, option
import numpy as np
from mmap import mmap, ACCESS_READ


HEADER_BUF_SIZE = 1024  # size of file header
DATA_BUF_SIZE = 16  # size of each data block
EPOCH = datetime(1900, 1, 1)  # reference epoch for logged minutes

@group()
def cli():
    """
    Working with FPOD acoustic devices.
    """
    pass

@cli.group()
def fpod():
    """
    FPOD file operations
    """
    pass


def bslice(buf: bytes, offset: int, size: int) -> bytes:
    """Return a slice of the buffer from the given offset and size."""
    if offset + size > len(buf):
        raise IndexError("Out of bounds")
    stop = offset + size
    return buf[offset:stop]


def int_from_bytes(buf: bytes, offset: int, size: int, signed: bool = False) -> int:
    """Return an integer from the given slice of the buffer."""
    return int.from_bytes(bslice(buf, offset, size), byteorder="big", signed=signed)


def string_from_bytes(buf: bytes, offset: int, size: int) -> str:
    """Return a string from the given slice of the buffer."""
    return bslice(buf, offset, size).decode("ascii").rstrip("\x00")


class Header():
    """
    Header information for an FPOD FP1 file.
    """
    pod_id: int
    first_logged_min: int
    last_logged_min: int
    water_depth: float
    deployment_depth: float
    lat_text: str
    lon_text: str
    location_text: str
    notes_text: str
    gmt_text: str
    pic_ver: int
    fpga_ver: int
    extended_amps: bool

    def __init__(self, buf: bytes):
        self.pod_id = 100 * buf[3] + buf[4]
        self.first_logged_min = int_from_bytes(buf, 256, 4, signed=True)
        self.last_logged_min = int_from_bytes(buf, 260, 4, signed=True)
        self.water_depth = (buf[131] << 8) + buf[132]
        self.deployment_depth = (buf[129] << 8) + buf[130]
        self.lat_text = string_from_bytes(buf, 133, 11)
        self.lon_text = string_from_bytes(buf, 145, 11)
        self.location_text = string_from_bytes(buf, 157, 30)
        self.notes_text = string_from_bytes(buf, 188, 43)
        self.gmt_text = string_from_bytes(buf, 232, 11)
        self.pic_ver = buf[37]
        self.fpga_ver = buf[39] << 8 | buf[40]
        self.extended_amps = self.fpga_ver > 0


class Message(IntEnum):
    """
    Message types for FPOD FP1 and FP3 files.
    """
    # From 0 through 183, click recorded
    CLICK = 183
    # Acoustic release channel
    ACOUSTIC_RELEASE = 245
    # Error reporting channel
    RUN_ERROR = 247
    # Used to interpret ordering of train messages
    FP3_VERSION = 248
    # train details, once clicks are classified, not in FP1 files
    TRAIN = 249
    WAV = 250
    # User discarded train data
    TRASH = 251
    SONAR_GHOST = 252
    # Referred to as "duff" in original implementation
    REJECT = 253
    TIMESTAMP = 254
    END_MARKER = 255
    OTHER = 256

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, int):
            if value <= 183:
                return cls.CLICK
            elif value == 250:
                return cls.WAV
            elif value == 254:
                return cls.TIMESTAMP
            else:
                return cls.OTHER
        return super()._missing_(value)


# Data row structure for environmental data, Big Endian
timestamp_row = np.dtype([
    ('message', np.uint8), # 0: message type
    ('blnk', np.uint8), # 1: blank
    ('blnk2', np.uint8), # 2: blank
    ('angle_x', np.uint8), # 3: angle
    ('angle_y', np.uint8), # 4: angle
    ('angle_z', np.uint8), # 5: angle
    ('blnk3', np.uint8), # 6: blank, used for FP3 files
    ('temp_deg_c', np.uint8), # 7: temperature in degrees Celsius
    ('blnk4', np.uint8), # 8: blank, used for FP3 files
    ('blnk5', np.uint8), # 9: blank
    ('flags', np.uint8), # 10: flags byte
    ('bat1', np.uint8), # 11: battery
    ('bat2', np.uint8), # 12: battery
    ('annotation_code', np.uint8), # 13: annotation code, FP3
    ('minute_deadband', np.uint8), # 14: minute deadband
    ('system_flags', np.uint8), # 15: system flags, unused
]).newbyteorder('>')

click_row = np.dtype([
    ('ts_hi', np.uint8), # 0: high byte of timestamp; also the click record type in 0..183
    ('ts_mid', np.uint8), # 1: middle byte of timestamp
    ('ts_lo', np.uint8), # 2: low byte of timestamp
    ('ncyc', np.uint8), # 3: number of cycles
    ('clk_ipi_range', np.uint8), # 4: compressed multiple values
    ('ipi_pre_max', np.uint8), # 5: inter-pulse interval
    ('ipi_at_max', np.uint8), # 6: inter-pulse interval at maximum
    ('ipi_plus_1', np.uint8), # 7: inter-pulse interval plus 1
    ('ipi_plus_2', np.uint8), # 8: blank, used for FP3 files
    ('raw_pk', np.uint8), # 9: raw peak value
    ('max_pk', np.uint8), # 10: raw peak max
    ('raw_pk_plus_1', np.uint8), # 11: raw peak plus 1
    ('ipi_before', np.uint8), # 12: inter-pulse interval before / QNN
    ('amp_rev_dur', np.uint8), # 13: amp reversals / duration
    ('duration_overflow', np.uint8), # 14: duration overflow
    ('sonar_found', np.uint8), # 15: sonar found
]).newbyteorder('>')


def read_env_messages(filepath: Path) -> DataFrame:
    with open(filepath, "rb") as fid:
        header = Header(fid.read(HEADER_BUF_SIZE))
        mm = mmap(fid.fileno(), 0, access=ACCESS_READ)
        nbytes = mm.size() - HEADER_BUF_SIZE
        raw = np.frombuffer(
            mm,
            dtype=timestamp_row,
            count = nbytes // DATA_BUF_SIZE,
            offset = HEADER_BUF_SIZE
        )
    # keep only selected records
    env_mask = raw["message"] == Message.TIMESTAMP
    df = DataFrame.from_records(raw[env_mask])
    start = EPOCH + timedelta(minutes=header.first_logged_min)
    df.index = Timestamp(start) + to_timedelta(np.arange(len(df)), unit="m")
    return df

def filled_timestamp_column(
    records: np.ndarray,
    first_logged_min: int
):
    msg = records['ts_hi']
    ts_mask = (msg == Message.TIMESTAMP)
    dts = np.full(len(records), np.datetime64("NaT"), dtype="datetime64[ns]")
    start = np.datetime64(EPOCH + timedelta(minutes=first_logged_min))
    ts_positions = np.flatnonzero(ts_mask)
    if len(ts_positions):
        dts[ts_positions] = start + np.arange(len(ts_positions), dtype="timedelta64[m]")
    return Series(dts).ffill()

def read_click_messages(filepath: Path) -> DataFrame:
    with open(filepath, "rb") as fid:
        header = Header(fid.read(HEADER_BUF_SIZE))
        mm = mmap(fid.fileno(), 0, access=ACCESS_READ)
        nbytes = mm.size() - HEADER_BUF_SIZE
        raw = np.frombuffer(
            mm,
            dtype=click_row,
            count = nbytes // DATA_BUF_SIZE,
            offset = HEADER_BUF_SIZE
        )
    # keep only selected records. For click records, the first byte is the high byte
    # of the 24-bit timestamp and also the record class in the 0..183 range.
    click_mask = raw["ts_hi"] <= Message.CLICK
    df = DataFrame.from_records(raw)
    df.index = filled_timestamp_column(raw, header.first_logged_min) + np.array(5 * (
        (df["ts_hi"].to_numpy(dtype=np.uint32) << 16)
        | (df["ts_mid"].to_numpy(dtype=np.uint32) << 8)
        | df["ts_lo"].to_numpy(dtype=np.uint32)
    ), dtype="timedelta64[us]")
    return df[click_mask]


@fpod.command("env")
@argument("filepath", type=Path, help="Path to the FPOD binary file")
def fpod_env(filepath: Path):
    
    start_time = datetime.now()
    df = read_env_messages(filepath)[['temp_deg_c', 'angle_x']]
    mask = df["angle_x"] < 5
    df = df[mask]
    df = cast(DataFrame, df)
    daily = df.resample("D", closed="left", label="left").mean()
    daily = cast(DataFrame, daily)
    fig, ax = subplots(figsize=(5, 3))
    ax.plot(daily.index, daily['temp_deg_c'], color='black')
    ax.tick_params("x", rotation=45, rotation_mode="xtick")
    fig.tight_layout()
    fig.savefig("temp_deg_c_plot.png", dpi=300, bbox_inches='tight')
       
    echo(f"Elapsed time: {datetime.now() - start_time}")

@fpod.command("clicks")
@argument("filepath", type=Path, help="Path to the FPOD binary file")
@option("--stop-after", type=int, help="Stop after reading this many records")
def fpod_clicks(filepath: Path, stop_after: Optional[int]):
    df = read_click_messages(filepath)[["ncyc"]]

    print(df.head(50))
    # print(df.describe())

if __name__ == "__main__":
    set_option('display.max_columns', None)
    cli()
