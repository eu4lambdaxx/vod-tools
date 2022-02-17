import cv2 # Assumes at least version 3
from functools import total_ordering
import numpy as np
import sys
import os
import pytesseract
import dateparser
import matplotlib.pyplot as plt
from spellchecker import SpellChecker

@total_ordering
class Eu4Date:
    mlimit = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    # more compactful
    def __init__(self, year, month, day):
        self.y = year
        self.m = month
        self.d = day

    def tomorrow(self):
        return Eu4Date(self.y, self.m, self.d + 1)

    # bad loops
    def normalize(self):
        while self.d < 1:
            self.d += self.mlimit[(self.m - 2)%12] + 1
            self.m -= 1
        while self.d > self.mlimit[(self.m - 1)%12]:
            self.d -= self.mlimit[(self.m - 1)%12]
            self.m += 1
        while self.m < 1:
            self.m += 12
            self.y -= 1
        while self.m > 12:
            self.m -= 12
            self.y += 1

    def days_since_epoch(self):
        epoch = Eu4Date(1444, 11, 11)
        y = self.y
        m = self.m
        d = self.d
        if m < epoch.m:
            m += 12
            y -= 1 
        days = 0
        days += (y-epoch.y)*365
        while epoch.m < m:
            days += self.mlimit[(epoch.m-1)%12]
            epoch.m += 1
        days += (d - epoch.d)
        return days

    def __eq__(self, obj):
        return isinstance(obj, Eu4Date) and self.y == obj.y and self.m == obj.m and self.d == obj.d

    def __lt__ (self, obj):
        if not isinstance(obj, Eu4Date):
            return False
        if self.y == obj.y:
            if self.m == obj.m:
                return self.d < obj.d
            else:
                return self.m < obj.m
        else:
            return self.y < obj.y




class Eu4vod:
    def __init__(self, vid_path):
        self.v = cv2.VideoCapture(vid_path)
        self.fps = self.v.get(cv2.CAP_PROP_FPS)
        self.w = self.v.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.h = self.v.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.sc = SpellChecker() # TODO, customize

    def __del__(self):
        self.v.release()
        cv2.destroyAllWindows()

    def in_game(self, frame):
        # technically date can be covered by event or errors can occur, so more metrics should be used...
        # GIVE ME MAYBE MONADS lambda11PsiRiot
        return self.get_date(frame) is not None

    def get_date(self, frame):
        # top right corner... depends on UI scaling and mods though :/
        x = int(self.w * 0.885)
        w = int(self.w * 0.065)
        y = int(self.h * 0.01)
        h = int(self.h * 0.03)
        date_frame = frame[y:y+h, x:x+w]
        date_frame = cv2.cvtColor(date_frame, cv2.COLOR_BGR2GRAY)
        date_frame = cv2.bitwise_not(date_frame)
        # upscaling doesn't seem to be necessary on 1080p but helps for 720p
        #date_frame = cv2.resize(date_frame, (0, 0), fx=2, fy=2)
        text = pytesseract.image_to_string(date_frame, config='--psm 6 --user-words words.txt --user-patterns patterns.txt')
        words = text.split(' ')
        date = ''
        for w in words:
            date += self.sc.correction(w) + ' '
        d = dateparser.parse(text)
        # assumes no mods, but one quick high probability way to remove bad matches
        if d is not None:
            if d.year < 1444 or d. year > 1821:
                return None
            else:
                return Eu4Date(d.year, d.month, d.day)
        else:
            return None

    def seek_time(self, seconds):
        return self.seek_frames(self.fps * seconds)

    def seek_frames(self, frames):
        i = 0;
        while i < frames and self.v.isOpened():
            ret, frame = self.v.read()
            i += 1
        return ret, frame

    def seek_until_date(self, date, step = 30):
        while True:
            ret, frame = self.seek_frames(step)
            if not ret:
                return ret, frame
            cv2.imshow('frame', frame)
            today = self.get_date(frame)
            if today > date:
                return ret, frame

    def interactive(self):
        while (self.v.isOpened()):
            ret, frame = self.v.read()
            if not ret:
                break
            cv2.imshow('frame', frame)
            k = cv2.waitKey(0) & 0xFF
            # shortcuts:
            #   q - quit
            #   m - skip one minute
            #   s - skip one second
            #   d - debug (custom interaction)

            if k == ord('q'):
                break
            elif k == ord('m'):
                self.seek_time(60)
            elif k == ord('s'):
                self.seek_time(1)
            elif k == ord('d'):
                d = self.get_date(frame)
                if d is not None:
                    self.seek_until_date(d)

        cv2.destroyAllWindows()

    def analytic(self, step = 1):
        # list of dates
        l = []
        while (self.v.isOpened()):
            ret, frame = self.v.read()
            if not ret:
                break
            d = self.get_date(frame)
            if d is None:
                print("No date")
                l.append(-1)
            else:
                days = d.days_since_epoch()
                print("Date: {}.{}.{}, Days: {}".format(d.y, d.m, d.d, days))
                l.append(days)
        return l



def main(vid_path):
    if not os.path.isfile(vid_path):
        print("Cannot open given video file")
    e = Eu4vod(vid_path)
    l = e.analytic()
    with open('result.txt', 'w') as f:
        for n in l:
            f.write(str(n) + "\n")

def plot():
    plt.ion()
    with open('result.txt', 'r') as f:
        data = [int(x) for x in f.readlines()]
    plt.plot(data, 'ro', ms=0.05)
    plt.ylabel('Days since 1444.11.11')
    plt.xlabel('Frame (1 corresponds to 1 second)')
    plt.savefig('result.png')
    plt.show(block=True)


if __name__ == '__main__':
    if sys.argv[1] == 'plot':
        plot()
    else:
        main(sys.argv[1])
