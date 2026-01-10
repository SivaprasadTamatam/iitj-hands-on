

import pandas as pd
import numpy as np
import random
import re
from datetime import datetime, timedelta
from IPython.display import display


random.seed(42)
np.random.seed(42)


INACTIVITY_MINUTES = 30


ALLOWED_EVENTS = ["landing","search","product_view","add_to_cart","checkout","payment","logout"]
ALLOWED_DEVICES = ["Android","iOS","Web"]
ALLOWED_CITIES = ["Bengaluru","Mumbai","Delhi","Pune","Chennai"]


print("Ready. INACTIVITY_MINUTES =", INACTIVITY_MINUTES)

