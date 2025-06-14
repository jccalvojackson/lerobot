import glob
import json
import os
import subprocess
from typing import Any, Dict, List, Optional

import serial.tools.list_ports


def find_ports_pyserial() -> List[Dict[str, Any]]:
    """Original method using pySerial's list_ports."""
    ports = []
    for port in serial.tools.list_ports.comports():
        ports.append(
            {
                "device": port.device,
                "description": port.description,
                "manufacturer": port.manufacturer,
                "product": port.product,
                "vid": port.vid,
                "pid": port.pid,
                "serial_number": port.serial_number,
                "location": getattr(port, "location", None),
                "interface": getattr(port, "interface", None),
                "hwid": port.hwid,
                "method": "pyserial",
            }
        )
    return ports


def find_ports_system_profiler() -> List[Dict[str, Any]]:
    """Use macOS system_profiler to find USB devices with serial capability."""
    try:
        result = subprocess.run(
            ["system_profiler", "-json", "SPUSBDataType"], capture_output=True, text=True, timeout=10
        )

        if result.returncode != 0:
            return []

        data = json.loads(result.stdout)
        usb_ports = []

        def extract_usb_info(items, parent_location=""):
            for item in items:
                # Check if this device might be a serial device
                name = item.get("_name", "")
                location_id = item.get("location_id", "")
                vendor_id = item.get("vendor_id", "")
                product_id = item.get("product_id", "")
                serial_num = item.get("serial_num", "")

                # Common indicators of serial devices
                serial_indicators = [
                    "serial",
                    "uart",
                    "usb",
                    "cp210",
                    "ft232",
                    "ch340",
                    "arduino",
                    "esp",
                    "converter",
                    "adapter",
                ]

                if any(indicator in name.lower() for indicator in serial_indicators):
                    # Try to find corresponding /dev entry
                    device_path = find_device_path_for_usb(vendor_id, product_id, serial_num)

                    usb_ports.append(
                        {
                            "device": device_path or f"USB:{location_id}",
                            "description": name,
                            "manufacturer": item.get("manufacturer", ""),
                            "product": name,
                            "vid": vendor_id,
                            "pid": product_id,
                            "serial_number": serial_num,
                            "location_id": location_id,
                            "method": "system_profiler",
                        }
                    )

                # Recursively check nested items
                if "_items" in item:
                    extract_usb_info(item["_items"], location_id)

        extract_usb_info(data.get("SPUSBDataType", []))
        return usb_ports

    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        print(f"Error with system_profiler: {e}")
        return []


def find_device_path_for_usb(vendor_id: str, product_id: str, serial_num: str) -> Optional[str]:
    """Try to find the /dev path for a USB device based on its properties."""
    # Check common serial device paths
    dev_paths = glob.glob("/dev/cu.*") + glob.glob("/dev/tty.*")

    for dev_path in dev_paths:
        try:
            # Use ioreg to get detailed info about this device
            result = subprocess.run(
                ["ioreg", "-c", "IOSerialBSDClient", "-r"], capture_output=True, text=True, timeout=5
            )

            if dev_path.split("/")[-1] in result.stdout:
                # This is a rough match - in a production system you'd want more precise matching
                return dev_path

        except Exception:
            continue

    return None


def find_ports_ioreg() -> List[Dict[str, Any]]:
    """Use macOS ioreg to find serial devices."""
    try:
        result = subprocess.run(
            ["ioreg", "-c", "IOSerialBSDClient", "-r"], capture_output=True, text=True, timeout=10
        )

        if result.returncode != 0:
            return []

        ports = []
        lines = result.stdout.split("\n")
        current_device = {}

        for line in lines:
            line = line.strip()
            if "IODialinDevice" in line:
                # Extract device path
                if '"' in line:
                    device = line.split('"')[1]
                    current_device["device"] = f"/dev/{device}"
            elif "IOCalloutDevice" in line:
                if '"' in line:
                    device = line.split('"')[1]
                    current_device["callout_device"] = f"/dev/{device}"
            elif "IOTTYBaseName" in line:
                if '"' in line:
                    base_name = line.split('"')[1]
                    current_device["base_name"] = base_name
            elif line.startswith("+") or line.startswith("|"):
                # End of current device block
                if current_device and "device" in current_device:
                    current_device["method"] = "ioreg"
                    current_device["description"] = current_device.get("base_name", "Unknown")
                    ports.append(current_device.copy())
                current_device = {}

        return ports

    except Exception as e:
        print(f"Error with ioreg: {e}")
        return []


def find_ports_filesystem() -> List[Dict[str, Any]]:
    """Find serial ports by scanning /dev directory."""
    ports = []

    # Get all potential serial device files
    cu_devices = glob.glob("/dev/cu.*")
    tty_devices = glob.glob("/dev/tty.*")

    # Filter out common non-serial devices
    excluded_patterns = ["Bluetooth", "bluetooth", "debug", "console"]

    all_devices = cu_devices + tty_devices

    for device in all_devices:
        device_name = os.path.basename(device)

        # Skip excluded patterns
        if any(pattern.lower() in device_name.lower() for pattern in excluded_patterns):
            continue

        # Check if device is accessible
        try:
            if os.path.exists(device):
                ports.append(
                    {
                        "device": device,
                        "description": f"Filesystem found: {device_name}",
                        "method": "filesystem",
                    }
                )
        except Exception:
            continue

    return ports


def find_all_ports() -> Dict[str, List[Dict[str, Any]]]:
    """Find ports using all available methods."""
    methods = {
        "pyserial": find_ports_pyserial,
        "system_profiler": find_ports_system_profiler,
        "ioreg": find_ports_ioreg,
        "filesystem": find_ports_filesystem,
    }

    results = {}
    for method_name, method_func in methods.items():
        print(f"Scanning with {method_name}...")
        try:
            results[method_name] = method_func()
        except Exception as e:
            print(f"Error with {method_name}: {e}")
            results[method_name] = []

    return results


def merge_and_deduplicate_ports(all_results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Merge results from different methods and remove duplicates."""
    seen_devices = set()
    merged_ports = []

    # Prioritize methods (system_profiler often has most complete info)
    priority_order = ["system_profiler", "ioreg", "pyserial", "filesystem"]

    for method in priority_order:
        if method in all_results:
            for port in all_results[method]:
                device = port.get("device", "")
                if device and device not in seen_devices:
                    seen_devices.add(device)
                    merged_ports.append(port)

    return merged_ports


def print_port_info(ports: List[Dict[str, Any]], title: str = "Serial Ports"):
    """Print formatted port information."""
    if not ports:
        print(f"No {title.lower()} found.")
        return

    print(f"\n{title} ({len(ports)} found):")
    print("=" * 50)

    for i, port in enumerate(ports, 1):
        print(f"{i}. Device: {port.get('device', 'Unknown')}")
        print(f"   Description: {port.get('description', 'N/A')}")
        print(f"   Method: {port.get('method', 'N/A')}")

        if port.get("manufacturer"):
            print(f"   Manufacturer: {port['manufacturer']}")
        if port.get("product"):
            print(f"   Product: {port['product']}")
        if port.get("vid"):
            print(f"   VID: {port['vid']}")
        if port.get("pid"):
            print(f"   PID: {port['pid']}")
        if port.get("serial_number"):
            print(f"   Serial Number: {port['serial_number']}")
        if port.get("location_id"):
            print(f"   Location ID: {port['location_id']}")

        print("-" * 30)


def find_mac_ports(verbose: bool = False):
    """Enhanced function to find available serial ports on macOS using multiple methods."""
    print("Enhanced macOS Serial Port Detection")
    print("====================================")

    # Get results from all methods
    all_results = find_all_ports()

    if verbose:
        # Show results from each method separately
        for method, ports in all_results.items():
            print_port_info(ports, f"{method.replace('_', ' ').title()} Results")

    # Merge and deduplicate
    merged_ports = merge_and_deduplicate_ports(all_results)
    print_port_info(merged_ports, "All Available Serial Ports")

    # Summary
    total_found = sum(len(ports) for ports in all_results.values())
    unique_found = len(merged_ports)

    print(f"\nSummary:")
    print(f"  Total detections: {total_found}")
    print(f"  Unique ports: {unique_found}")
    print(f"  Methods used: {len([m for m, p in all_results.items() if p])}")

    if not merged_ports:
        print("\nTroubleshooting tips:")
        print("1. Make sure USB drivers are installed for your devices")
        print("2. Try unplugging and reconnecting USB devices")
        print("3. Check System Information > USB for connected devices")
        print("4. Some devices may require specific drivers (CP210x, FTDI, etc.)")

    return merged_ports


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Find serial ports on macOS")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show results from each method separately"
    )
    args = parser.parse_args()

    find_mac_ports(verbose=args.verbose)
    find_mac_ports(verbose=args.verbose)
