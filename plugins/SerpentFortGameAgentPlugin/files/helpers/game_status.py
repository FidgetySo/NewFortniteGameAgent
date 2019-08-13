from .memreader import MemoryReader

class Game:
    def __init__(self, **kwargs):
        self.reader = MemoryReader("ARCADE GAME SERIES: PAC-MAN")
        
        health_path = ["Release_3.dll", 0x3C440, 0x170, 0x8, 0x20, 0x0, 0x18]
        self.reader.store_address("health", health_path)

        score_path = ["tier0_s64.dll", 0x169FB8, 0x110, 0x0, 0x258]
        self.reader.store_address("score", score_path)

        paused_path = ["mono.dll", 0x264140, 0x110, 0x80, 0xA0, 0x398, 0x98]
        self.reader.store_address("paused", paused_path)

        game_over_path = ["mono.dll", 0x260110, 0x178, 0x5C]
        self.reader.store_address("game_over", game_over_path)
        

    def GetLives(self):
        lives = self.reader.read("health")

        return {
            1077952576: 1,
            555761728: 2,
            555753760: 3
        }.get(lives, 4)

    def GetScore(self):
        return self.reader.read("score")

    def IsPaused(self):
        paused = self.reader.read("paused")

        return {
            0: True,
            1: False
        }.get(paused, False)

    def IsOver(self):
        game_over = self.reader.read("game_over")
        
        return {
            0: True,
            1: False
        }.get(game_over, False)