"""Record which machine and software produced a result.

Every result Brokkr saves gets one of these attached, so any number can be traced
back to the exact CPU, OS, library versions, and code version that produced it.
This module has no Brokkr-specific assumptions and uses only the standard library.
"""

import datetime
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

DEFAULT_PACKAGES = ("brokkr", "numpy", "torch", "torchvision", "onnx", "onnxruntime")


def cpu_model() -> str:
    """Return a human-readable CPU name, e.g. 'Intel(R) Core(TM) i5-1135G7 @ 2.40GHz'."""
    system = platform.system()
    try:
        if system == "Windows":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        if system == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if system == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
    except OSError:
        pass
    # Fallback, e.g. ARM Linux boards whose /proc/cpuinfo has no "model name" line.
    return platform.processor() or platform.machine() or "unknown"


def board_model() -> str | None:
    """Return the board name on single-board computers (e.g. 'Raspberry Pi 5 Model B'), else None."""
    try:
        with open("/proc/device-tree/model", encoding="utf-8") as f:
            return f.read().strip("\x00\n ")
    except OSError:
        return None


def package_versions(packages=DEFAULT_PACKAGES) -> dict:
    """Return {package: version}, with None for packages that are not installed."""
    versions = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_info() -> dict:
    """Return the current git commit and whether there are uncommitted changes.

    'dirty' = True means the code that ran is not exactly the code in that commit.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


# Windows 10/11 power mode slider ("Best power efficiency" ... "Best performance").
WINDOWS_POWER_MODES = {
    "961cc777-2547-4f9d-8174-7d86181b8a7a": "best power efficiency",
    "00000000-0000-0000-0000-000000000000": "balanced",
    "ded574b5-45a0-4f42-8737-46345c09c238": "best performance",
}


def power_state() -> dict:
    """Is the machine plugged in, and (on Windows) which power mode is it in?

    Laptops on battery or in power-saving modes run slower, so speed results must record this.
    None means "could not tell" (e.g. a desktop or board with no battery information).
    """
    state = {"on_ac_power": None, "battery_percent": None, "battery_saver": None, "power_mode": None}
    if platform.system() == "Windows":
        import ctypes

        class SystemPowerStatus(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                        ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                        ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]

        status = SystemPowerStatus()
        if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
            if status.ACLineStatus in (0, 1):  # 255 = unknown
                state["on_ac_power"] = bool(status.ACLineStatus)
            if status.BatteryLifePercent <= 100:  # 255 = unknown
                state["battery_percent"] = status.BatteryLifePercent
            state["battery_saver"] = bool(status.SystemStatusFlag)

        class Guid(ctypes.Structure):
            _fields_ = [("d1", ctypes.c_ulong), ("d2", ctypes.c_ushort), ("d3", ctypes.c_ushort),
                        ("d4", ctypes.c_ubyte * 8)]

        guid = Guid()
        try:
            if ctypes.windll.powrprof.PowerGetEffectiveOverlayScheme(ctypes.byref(guid)) == 0:
                d4 = bytes(guid.d4).hex()
                text = f"{guid.d1:08x}-{guid.d2:04x}-{guid.d3:04x}-{d4[:4]}-{d4[4:]}"
                state["power_mode"] = WINDOWS_POWER_MODES.get(text, text)
        except (AttributeError, OSError):  # older Windows versions lack this function
            pass
    elif platform.system() == "Linux":
        # Mains adapters appear as /sys/class/power_supply/<name>/ with type "Mains".
        for supply in Path("/sys/class/power_supply").glob("*"):
            try:
                if (supply / "type").read_text().strip() == "Mains":
                    state["on_ac_power"] = (supply / "online").read_text().strip() == "1"
                elif (supply / "type").read_text().strip() == "Battery":
                    state["battery_percent"] = int((supply / "capacity").read_text())
            except (OSError, ValueError):
                pass
    return state


def os_release() -> str:
    """Return the OS release, e.g. '11' for Windows 11.

    Python before 3.12 reports Windows 11 as '10'; Windows 11 is build 22000 or later.
    """
    release = platform.release()
    if platform.system() == "Windows" and release == "10":
        build = int(platform.version().split(".")[-1])
        if build >= 22000:
            return "11"
    return release


def machine_fingerprint(packages=DEFAULT_PACKAGES) -> dict:
    """Return a JSON-serialisable description of this machine and software environment."""
    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "cpu_model": cpu_model(),
        "cpu_count_logical": os.cpu_count(),
        "board_model": board_model(),
        "architecture": platform.machine(),
        "os": platform.system(),
        "os_release": os_release(),
        "os_version": platform.version(),
        "power": power_state(),
        "python_version": sys.version.split()[0],
        "packages": package_versions(packages),
        "git": git_info(),
    }
