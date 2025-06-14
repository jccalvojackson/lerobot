#!/usr/bin/env python3
"""
Simple script to test if a specific port can communicate with a MotorsBus device.
This is useful when the automatic port detection doesn't work due to USB hubs.
"""

import argparse
import sys
import time

try:
    import serial
except ImportError:
    print("Error: pyserial not installed. Install with: pip install pyserial")
    sys.exit(1)


def test_port(port_path, timeout=2.0):
    """
    Test if a port can be opened and potentially communicate with a MotorsBus.

    Args:
        port_path (str): Path to the serial port (e.g., '/dev/cu.usbmodemSN234567892')
        timeout (float): Connection timeout in seconds

    Returns:
        bool: True if port seems to work, False otherwise
    """
    print(f"Testing port: {port_path}")

    try:
        # Try to open the serial port with common MotorsBus settings
        # Common baud rates for motor buses: 115200, 57600, 9600, 1000000
        baud_rates = [115200, 57600, 1000000, 9600]

        for baud_rate in baud_rates:
            print(f"  Trying baud rate: {baud_rate}")
            try:
                with serial.Serial(port_path, baud_rate, timeout=timeout) as ser:
                    print(f"    ✅ Successfully opened at {baud_rate} baud")

                    # Try to read/write (basic test)
                    ser.flushInput()
                    ser.flushOutput()

                    # Check if we can read anything (don't expect specific response)
                    time.sleep(0.1)
                    if ser.in_waiting > 0:
                        data = ser.read(ser.in_waiting)
                        print(f"    📨 Received {len(data)} bytes: {data}")

                    print(f"    ✅ Port {port_path} appears to be working at {baud_rate} baud")
                    return True

            except serial.SerialException as e:
                print(f"    ❌ Failed at {baud_rate} baud: {e}")
                continue

        print(f"  ❌ Could not connect at any standard baud rate")
        return False

    except Exception as e:
        print(f"  ❌ Error testing port: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test a serial port for MotorsBus communication")
    parser.add_argument("port", help="Serial port path (e.g., /dev/cu.usbmodemSN234567892)")
    parser.add_argument("--timeout", type=float, default=2.0, help="Connection timeout (default: 2.0s)")

    args = parser.parse_args()

    print("MotorsBus Port Tester")
    print("=" * 40)
    print(f"Target port: {args.port}")
    print(f"Timeout: {args.timeout}s")
    print()

    # Test the port
    success = test_port(args.port, args.timeout)

    print()
    if success:
        print(f"🎉 SUCCESS: Port {args.port} appears to be working!")
        print(f"You can use this port in your MotorsBus configuration:")
        print(f'  port="{args.port}"')
    else:
        print(f"❌ FAILED: Port {args.port} does not seem to be working.")
        print("Possible issues:")
        print("1. Wrong port path")
        print("2. Device not connected or powered")
        print("3. Device using non-standard baud rate")
        print("4. Permission issues (try with sudo on Linux)")
        print("5. Device requires specific initialization sequence")


if __name__ == "__main__":
    main()
