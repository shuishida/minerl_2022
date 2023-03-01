from datetime import datetime


def curr_time_str():
    return datetime.now().strftime('%m%d_%H%M%S')
