"""
Read FPOD FP1 file format
"""
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
from enum import IntEnum
from pandas import DataFrame
import numpy as np

DATAFILE = Path('/Users/keeney/Library/CloudStorage/GoogleDrive-nkeeney@hurricaneisland.net/Shared drives/Island Research /Projects & Focus Areas /2026_2025 UMaine Ocean and Climate Field Studies/Data/fpod_raw/Hurricane2026 2026 05 10 FPOD_6583 file0.FP1')

HEADER_BUF_SIZE = 1024
DATA_BUF_SIZE = 16

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



class Data():
    """
    Size of vectors is max clicks
    """
    # click data
    min: list[int]
    microsec: list[int]
    click_no: list[int]
    ncyc: list[int]
    pkat: list[int]
    clk_ipi_range: list[int]
    ipi_pre_max: list[int]
    ipi_at_max: list[int]
    khz: list[int]
    amp_at_max: list[int]
    amp_reversals: list[int]
    duration: list[float]
    has_wav: list[bool]

    # environmental data
    bat_use: list[int]
    prior_min: list[bool]
    next_min: list[bool]

    def __init__(self):
        self.min = []
        self.microsec = []
        self.click_no = []
        self.ncyc = []
        self.pkat = []
        self.clk_ipi_range = []
        self.ipi_pre_max = []
        self.ipi_at_max = []
        self.khz = []
        self.amp_at_max = []
        self.amp_reversals = []
        self.duration = []
        self.has_wav = []

        self.bat_use = []
        self.prior_min = []
        self.next_min = []

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



def decode_header(buf: bytes) -> Header:
    header = Header()
    header.pod_id = 100 * buf[3] + buf[4]
    header.first_logged_min = int_from_bytes(buf, 256, 4, signed=True)
    header.last_logged_min = int_from_bytes(buf, 260, 4, signed=True)
    header.water_depth = (buf[131] << 8) + buf[132]
    header.deployment_depth = (buf[129] << 8) + buf[130]
    header.lat_text = string_from_bytes(buf, 133, 11)
    header.lon_text = string_from_bytes(buf, 145, 11)
    header.location_text = string_from_bytes(buf, 157, 30)
    header.notes_text = string_from_bytes(buf, 188, 43)
    header.gmt_text = string_from_bytes(buf, 232, 11)
    header.pic_ver = buf[37]
    header.fpga_ver = buf[39] << 8 | buf[40]
    header.extended_amps = header.fpga_ver > 0
    return header

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


def read_file(filepath: Path) -> Dict[int, int]:
    basename = filepath.stem
    ext = filepath.suffix
    file_size = filepath.stat().st_size
    max_rows = (file_size - HEADER_BUF_SIZE) // DATA_BUF_SIZE
    messages: Dict[int, int] = {}
    env_data: List[np.ndarray] = []
    epoch = datetime(1900, 1, 1)

    with open(filepath, 'rb') as fid:
        header_bytes = fid.read(HEADER_BUF_SIZE)
        header = decode_header(header_bytes)
        for attr, value in vars(header).items():
            print(f"{attr}: {value}")
        current_min = -1
        start = epoch + timedelta(minutes=header.first_logged_min)
        end = epoch + timedelta(minutes=header.last_logged_min)
        duration = header.last_logged_min - header.first_logged_min
        print(f"Start: {start}, End: {end}, Duration: {duration}")

        stop_after = 1000

        while(True):
            buf = fid.read(DATA_BUF_SIZE)
            if not buf or (len(buf) < DATA_BUF_SIZE):
                break  # partial record / eof
            msg = Message(buf[0])
            if msg not in messages:
                messages[msg] = 1
            else:
                messages[msg] += 1
            if msg == Message.ENV:
                current_min += 1
                row = np.frombuffer(buf, dtype=row_type)
                env_data.append(row)
                # flags_byte = buf[10]
                # data.bat_use.append((flags_byte & 2) + 1)
                # data.prior_min.append(bool(flags_byte & 1))
                # data.next_min.append(bool((flags_byte >> 2) & 1))
        print(DataFrame(env_data))
        return messages


if __name__ == "__main__":
    counts = read_file(Path(DATAFILE))
    print(counts)
    # print(json.dumps(sorted(counts.items()), indent=4))
    # print(len(counts))
