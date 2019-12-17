import sys

if sys.version_info > (2, 7):
    import math
    NaN = math.nan
else:
    NaN = float('nan')
