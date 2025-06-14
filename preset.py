# ezviz_goto_preset.py

from pyezviz.client import EzvizClient


# EZVIZ could user creds
EMAIL = ""
PASSWORD = ""
REGION = "sgp"
PRESET = "1"

client = EzvizClient(EMAIL, PASSWORD, REGION)
client.login()

devices = client.load_cameras()
if not devices:
    print("No devices found.")
    exit()

device_serial = list(devices.keys())[0]
print(f"Using camera: {device_serial}")

client.ptz_control_coordinates(device_serial, x_axis=0.46, y_axis=0.3) # CHANGE VALUES TO PERFECT YOUR CAMERA POSITION
#help(client.ptz_control)

print(f"Moved camera to preset {PRESET}")
