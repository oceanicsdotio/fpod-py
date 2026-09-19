"""
Read FPOD FP1 file format
"""
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, cast
from enum import IntEnum
from pandas import DataFrame, Timestamp, to_timedelta
import numpy as np
import mmap


HEADER_BUF_SIZE = 1024  # size of file header
DATA_BUF_SIZE = 16  # size of each data block
EPOCH = datetime(1900, 1, 1)  # reference epoch for logged minutes

def bslice(buf: bytes, offset: int, size: int) -> bytes:
    if offset + size > len(buf):
        raise IndexError("Out of bounds")
    stop = offset + size
    return buf[offset:stop]


def int_from_bytes(buf: bytes, offset: int, size: int, signed: bool = False) -> int:
    return int.from_bytes(bslice(buf, offset, size), byteorder="big", signed=signed)


def string_from_bytes(buf: bytes, offset: int, size: int) -> str:
    return bslice(buf, offset, size).decode("ascii").rstrip("\x00")


class Header():
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
    CLICK = 184
    TRAIN = 249
    WAV = 250
    ENV = 254
    OTHER = 255

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, int):
            if value < 184:
                return cls.CLICK
            elif value == 249:
                return cls.TRAIN
            elif value == 250:
                return cls.WAV
            elif value == 254:
                return cls.ENV
            else:
                return cls.OTHER
        return super()._missing_(value)


# Data row structure for environmental data, Big Endian
row_type = np.dtype([
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
    ('bat_use', np.uint8), # 10: flags byte
    ('bat1', np.uint8), # 11: battery
    ('bat2', np.uint8), # 12: battery
    ('annotation_code', np.uint8), # 13: annotation code, FP3
    ('minute_deadband', np.uint8), # 14: minute deadband
    ('system_flags', np.uint8), # 15: system flags, unused
]).newbyteorder('>')

def read_env_messages(filepath: Path, stop_after: Optional[int]) -> DataFrame:
    with open(filepath, "rb") as fid:
        header = Header(fid.read(HEADER_BUF_SIZE))
        start = EPOCH + timedelta(minutes=header.first_logged_min)
        mm = mmap.mmap(fid.fileno(), 0, access=mmap.ACCESS_READ)

        # slice off the header, then treat the rest as fixed-size records
        raw = np.frombuffer(mm[HEADER_BUF_SIZE:], dtype=np.uint8)
        records = raw[: len(raw) - (len(raw) % DATA_BUF_SIZE)]
        records = records.reshape(-1, DATA_BUF_SIZE)

        # keep only ENV records
        env_mask = records[:, 0] == Message.ENV
        env_records = records[env_mask]

        if stop_after is not None:
            env_records = env_records[:stop_after]

        # convert to structured dtype without Python row loop
        data = np.frombuffer(env_records.tobytes(), dtype=row_type)

        df = cast(DataFrame, DataFrame.from_records(data))
        df.index = Timestamp(start) + to_timedelta(np.arange(len(df)), unit="m")
        return df.resample("1h", closed="left", label="left").mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read an FPOD FP1 file.")
    parser.add_argument("filepath", type=Path, help="Path to the FPOD binary file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start_time = datetime.now()
    df = read_env_messages(args.filepath, stop_after=None)
    print(df.describe())
    print(f"Elapsed time: {datetime.now() - start_time}")
