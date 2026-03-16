# Model
MODEL_NAME = "yolov8l.pt"
CONFIDENCE_THRESHOLD = 0.10
IOU_THRESHOLD = 0.45
INFERENCE_IMG_SIZE = 1280

# COCO class IDs
PERSON_CLASS_ID = 0
BAG_CLASS_IDS = {24, 26, 28}  # backpack, handbag, suitcase

# Tracking
TRACKER_TYPE = "botsort_custom.yaml"
TRACK_HISTORY_LENGTH = 60

# Phantom bags (persist stationary bags after detection is lost)
BAG_STATIONARY_THRESH = 5  # frames before considered stationary
BAG_STATIONARY_MAX_DRIFT = 15  # pixels
BAG_PHANTOM_TTL = 60  # frames

# Association
PROXIMITY_MAX_DISTANCE = 200  # pixels
ASSOCIATION_MIN_SCORE = 0.15
OWNER_SWITCH_HYSTERESIS = 0.3

# Unattended logic
SEPARATION_DISTANCE = 150  # pixels
UNATTENDED_TIMEOUT = 3.0  # seconds

# Visualization (BGR)
COLOR_PERSON = (0, 200, 0)
COLOR_OWNED = (200, 150, 0)
COLOR_SEPARATED = (0, 220, 220)
COLOR_UNATTENDED = (0, 0, 220)
FONT_SCALE = 0.55
LINE_THICKNESS = 2
