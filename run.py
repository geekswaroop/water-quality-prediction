import os

f = open("res.txt", "a")
for classifier in range(1, 5):
    for epoch in range(1000, 10001, 1000):
        for rate in [0.01, 0.05, 0.1]:
            os.system("python3 main.py -model classifier{} -e {} -lr {} -tts 0.45".format(classifier, epoch, rate))