# Temporary cgi module patch for Python 3.13
def parse_header(line):
    parts = line.split(";", 1)
    key = parts[0].strip()
    pdict = {}
    if len(parts) > 1:
        for param in parts[1].split(";"):
            if "=" in param:
                k, v = param.strip().split("=", 1)
                pdict[k.strip()] = v.strip()
    return key, pdict