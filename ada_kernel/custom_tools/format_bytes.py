import math

def format_bytes(bytes: int, decimals: int = 1) -> str:
    if bytes == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = int(math.floor(math.log(bytes, 1024)))
    i = min(i, len(units) - 1)
    value = bytes / (1024 ** i)
    return f"{value:.{decimals}f} {units[i]}"