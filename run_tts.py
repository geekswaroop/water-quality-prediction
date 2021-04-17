import os

for tts in range (10, 51):
    os.system("python3 main.py -e 5000 -model classifier1 -tts {}".format(tts/100))