import datetime
import hashlib

#  Counters and hashing functions
counter = 0


def get_random_sha1() -> str:
    """Create image name"""
    global counter
    m = hashlib.sha1()
    m.update(str(counter).encode("ASCII"))
    m.update(str(datetime.datetime.now().timestamp()).encode("ASCII"))
    counter += 1
    return m.hexdigest()
