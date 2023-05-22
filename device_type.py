import sounddevice as sd

devices = sd.query_devices()
for i, device in enumerate(devices):
    print(f"Device #{i}: {device['name']}")
