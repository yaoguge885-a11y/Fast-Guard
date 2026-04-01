import threading
import winsound
import time

class CollisionAlarm:
    def __init__(self, frequency=2500, duration=80, interval=0.05):
        """
        :param frequency: 蜂鸣器频率(Hz)
        :param duration: 每次蜂鸣持续时间(ms)
        :param interval: 蜂鸣间隔(s)
        """
        self.freq = frequency
        self.duration = duration
        self.interval = interval
        
        self._alarming = threading.Event()
        self._stop_thread = False  
        
        self._thread = threading.Thread(target=self._alarm_loop, daemon=True)
        self._thread.start()

    def _alarm_loop(self):
        while not self._stop_thread:
            if self._alarming.is_set():
                winsound.Beep(self.freq, self.duration)
                time.sleep(self.interval)
            else:
                time.sleep(0.05)

    def trigger(self):
        if not self._alarming.is_set():
            self._alarming.set()

    def cease(self):
        if self._alarming.is_set():
            self._alarming.clear()

    def destroy(self):
        self.cease()
        self._stop_thread = True
