import re
from hashlib import sha1

# uid is hexdigest of sha1 value, which is double the digest size due to hex encoding
hash_len = sha1().digest_size * 2
uid_regex = re.compile("^[0-9a-f]{{{}}}$".format(hash_len), re.IGNORECASE)


def parse_reference(reference: str):
    tag = None
    uid = uid_regex.findall(reference)
    if not uid:
        tag = reference
    return tag, uid
