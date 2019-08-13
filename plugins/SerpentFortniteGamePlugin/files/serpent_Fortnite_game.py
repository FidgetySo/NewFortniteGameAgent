from serpent.game import Game

from .api.api import FortniteAPI

from serpent.utilities import Singleton

class SerpentFortniteGame(Game, metaclass=Singleton):

    def __init__(self, **kwargs):
        kwargs["platform"] = "executable"

        kwargs["window_name"] = "Fortnite  " # Do Not Change

        kwargs["executable_path"] = "C:/Program Files/Epic Games/Fortnite/FortniteGame/Binaries/Win64/FortniteClient-Win64-Shipping.exe"

        super().__init__(**kwargs)

        self.api_class = FortniteAPI
        self.api_instance = None

        self.environments = dict()
        self.environment_data = dict()
        self.frame_transformation_pipeline_string = "RESIZE:100x100|GRAYSCALE|FLOAT"

        self.frame_width = 100
        self.frame_height = 100
        self.frame_channels = 0
    @property
    def screen_regions(self):
        regions = {
            "HP_AREA": (140, 0, 480, 500),
            "SCORE_AREA": (40, 0, 80, 140),
            "GAME_REGION": (0, 0, 1080, 1920)
        }

        return regions

    @property
    def ocr_presets(self):
        presets = {
            "SAMPLE_PRESET": {
                "extract": {
                    "gradient_size": 1,
                    "closing_size": 1
                },
                "perform": {
                    "scale": 10,
                    "order": 1,
                    "horizontal_closing": 1,
                    "vertical_closing": 1
                }
            }
        }

        return presets
