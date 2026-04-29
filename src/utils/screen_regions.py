from dataclasses import dataclass


REFERENCE_WIDTH = 1920
REFERENCE_HEIGHT = 1080

# These preserve the old 1920x1080 crop: frame[30:800, 30:600].
HUD_LEFT = 30
HUD_TOP = 30
HUD_RIGHT = 600
HUD_BOTTOM = 800


@dataclass(frozen=True)
class ScreenRegion:
    left: int
    top: int
    right: int
    bottom: int
    scale: float

    @property
    def width(self):
        return self.right - self.left

    @property
    def height(self):
        return self.bottom - self.top


def _clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def get_resolution_scale(frame):
    height, width = frame.shape[:2]
    return max(0.1, min(width / REFERENCE_WIDTH, height / REFERENCE_HEIGHT))


def scale_pixels(value, scale, minimum=1):
    return max(minimum, int(round(value * scale)))


def get_hud_region(frame):
    height, width = frame.shape[:2]
    scale = get_resolution_scale(frame)

    left = _clamp(round(HUD_LEFT * scale), 0, max(0, width - 1))
    top = _clamp(round(HUD_TOP * scale), 0, max(0, height - 1))
    right = _clamp(round(HUD_RIGHT * scale), left + 1, width)
    bottom = _clamp(round(HUD_BOTTOM * scale), top + 1, height)

    region = ScreenRegion(
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        scale=scale,
    )
    return frame[region.top:region.bottom, region.left:region.right], region
