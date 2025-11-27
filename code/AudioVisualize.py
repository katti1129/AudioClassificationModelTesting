"""
  Index: 0, Name: ひとみ婆3のマイク
  Index: 1, Name: ひとみ婆さん’s AirPods Pro
  Index: 4, Name: ReSpeaker 4 Mic Array (UAC1.0)
  Index: 5, Name: MacBook Proのマイク
  Index: 7, Name: Microsoft Teams Audio

"""


import pyaudio

p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')

print("利用可能な入力デバイス:")
for i in range(0, numdevices):
    if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
        print(f"  Index: {i}, Name: {p.get_device_info_by_host_api_device_index(0, i).get('name')}")

p.terminate()