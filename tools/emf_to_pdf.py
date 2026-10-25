"""Extract a PDF embedded in an EMF file (as written by PowerPoint when a PDF is pasted).

Usage:
    python emf_to_pdf.py figure.emf figure.pdf
"""
import struct
import sys

b = open(sys.argv[1], 'rb').read()
i, chunks = 0, []
while i < len(b):
    rtype, size = struct.unpack_from('<II', b, i)
    if size == 0:
        break
    if rtype == 70:  # EMR_COMMENT
        n = struct.unpack_from('<I', b, i + 8)[0]
        chunks.append(b[i + 12:i + 12 + n])
    i += size
data = b''.join(chunks)
start, end = data.find(b'%PDF'), data.rfind(b'%%EOF')
if start < 0 or end < 0:
    sys.exit('no embedded PDF found')
open(sys.argv[2], 'wb').write(data[start:end + 5])
