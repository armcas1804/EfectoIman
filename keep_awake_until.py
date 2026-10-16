"""Prevent idle sleep temporarily while a scheduled local market capture runs.

Usage: python keep_awake_until.py 2026-10-13T06:40:00-07:00
Only system idle sleep is inhibited. The display may turn off normally.
"""

import argparse
import ctypes
import sys
import time
from datetime import datetime


ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


class SystemPowerStatus(ctypes.Structure):
    _fields_ = [
        ("ac_line_status", ctypes.c_ubyte),
        ("battery_flag", ctypes.c_ubyte),
        ("battery_life_percent", ctypes.c_ubyte),
        ("system_status_flag", ctypes.c_ubyte),
        ("battery_life_time", ctypes.c_ulong),
        ("battery_full_life_time", ctypes.c_ulong),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Keep Windows awake until a local deadline")
    parser.add_argument("until", help="Local ISO timestamp including UTC offset")
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("This helper requires Windows")
    deadline = datetime.fromisoformat(args.until)
    if deadline.tzinfo is None:
        parser.error("The deadline must include a UTC offset")

    kernel32 = ctypes.windll.kernel32
    kernel32.GetSystemPowerStatus.argtypes = [ctypes.POINTER(SystemPowerStatus)]
    kernel32.GetSystemPowerStatus.restype = ctypes.c_int
    kernel32.SetThreadExecutionState.argtypes = [ctypes.c_uint]
    kernel32.SetThreadExecutionState.restype = ctypes.c_uint

    print(f"Keeping system awake until {deadline.isoformat()} while connected to AC", flush=True)
    try:
        while (remaining := (deadline - datetime.now().astimezone()).total_seconds()) > 0:
            power = SystemPowerStatus()
            if not kernel32.GetSystemPowerStatus(ctypes.byref(power)):
                raise OSError("Cannot read power status")
            if power.ac_line_status != 1:
                print("AC power is disconnected; stopping the temporary wake guard", flush=True)
                return 1
            if not kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED):
                raise OSError("Cannot inhibit idle sleep")
            time.sleep(min(60, remaining))
        print("Deadline reached; restoring normal idle sleep", flush=True)
        return 0
    finally:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)


if __name__ == "__main__":
    raise SystemExit(main())
