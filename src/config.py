# src/config.py

X_CSV_FILE = 'x.csv'
Y_CSV_FILE = 'y.csv'
MODEL_RESULTS_CSV = 'model_results.csv'
DT_IMPORTANCE_PNG = 'top_10_features_dt.png'
CORRELATION_MAP_PNG = 'correlation_map.png'
TRAVEL_MODES_BAR_PNG = 'travel_modes_by_area.png'

HHPUB_CSV = 'hhpub.csv'
PERPUB_CSV = 'perpub.csv'
VEHPUB_CSV = 'vehpub.csv'
TRIPPUB_CSV = 'trippub.csv'

COLS_TO_ENCODE = [
    'WHYTO', 'WHYFROM', 'URBRUR', 'URBAN', 'HBHUR', 'CENSUS_R', 
    'R_SEX', 'EDUC', 'WORKER', 'LIF_CYC', 'MEDCOND', 'CONDRIVE', 
    'W_CHAIR', 'W_NONE', 'DRIVER', 'TRPHHVEH', 'TDWKND', 'LOOP_TRIP'
]
MIN_CLASS_COUNT = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42

LOGISTIC_MAX_ITER = 1000
DT_MAX_DEPTH = 20
DT_MIN_SAMPLES_SPLIT = 50

ACS_YEAR = 2024
ACS_SURVEY = "acs1"
ACS_GROUP = "S1901"
ACS_UCGID = "0100000US"
