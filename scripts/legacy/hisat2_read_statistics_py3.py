#!/usr/bin/env python3
# Python 3 port of HISAT2 2.2.0 hisat2_read_statistics.py (same stdout format for the wrapper).
# Drop-in replacement when python2 is unavailable.

import bz2
import gzip
import os
import sys
from argparse import ArgumentParser

COMPRESSION_NON = 0
COMPRESSION_GZIP = 1
COMPRESSION_BZIP2 = 2

SEQUENCE_FASTA = 0
SEQUENCE_FASTQ = 1


def parser_FQ(fp):
    while True:
        line = fp.readline()
        if line == "":
            return
        if line[0] == "@":
            break

    while True:
        rid = line[1:].split()[0]
        seq = ""

        line = fp.readline()
        if line == "":
            return

        seq = line.strip()
        yield rid, seq

        line = fp.readline()  # '+'
        line = fp.readline()  # quality
        line = fp.readline()  # next ID
        if line == "":
            return


def parser_FA(fp):
    while True:
        line = fp.readline()
        if line == "":
            return
        if line[0] == ">":
            break

    while True:
        rid = line[1:].split()[0]
        seq = ""

        while True:
            line = fp.readline()
            if line == "":
                break
            if line[0] == ">":
                break
            seq += line.strip()

        yield rid, seq

        if line == "":
            return


def parse_type(fname):
    compression_type = COMPRESSION_NON
    sequence_type = SEQUENCE_FASTA

    ff = fname.split(".")

    ext = ff[-1]
    if ext.lower() == "gz":
        compression_type = COMPRESSION_GZIP
        ext = ff[-2]
    elif ext.lower() == "bz2":
        compression_type = COMPRESSION_BZIP2
        ext = ff[-2]

    if ext.lower() in ("fq", "fastq"):
        sequence_type = SEQUENCE_FASTQ

    return sequence_type, compression_type


def generate_stats(length_map):
    mn = 0
    mx = 0
    cnt = 0
    avg = 0
    sum_len = 0

    if len(length_map) == 0:
        return cnt, mn, mx, avg

    sorted_map = sorted(length_map)

    mn = sorted_map[0]
    mx = sorted_map[-1]

    for k in sorted(length_map):
        sum_len += int(k) * length_map[k]
        cnt += length_map[k]

    avg = sum_len // cnt if cnt else 0

    return cnt, mn, mx, avg


def reads_stat(read_file, read_count):
    sequence_type, compression_type = parse_type(read_file)

    if compression_type == COMPRESSION_GZIP:
        fp = gzip.open(read_file, "rt", encoding="utf-8", errors="replace")
    elif compression_type == COMPRESSION_BZIP2:
        fp = bz2.open(read_file, "rt", encoding="utf-8", errors="replace")
    else:
        assert compression_type == COMPRESSION_NON
        fp = open(read_file, "r", encoding="utf-8", errors="replace")

    if sequence_type == SEQUENCE_FASTA:
        fstream = parser_FA(fp)
    else:
        assert sequence_type == SEQUENCE_FASTQ
        fstream = parser_FQ(fp)

    length_map = {}

    cnt = 0
    for _rid, seq in fstream:
        l = len(seq)
        length_map[l] = length_map.get(l, 0) + 1

        cnt += 1
        if read_count > 0 and cnt >= read_count:
            break

    fp.close()

    cnt, mn, mx, avg = generate_stats(length_map)
    # Match HISAT2 2.2.0 Py2: (k,v) = (read length, count); key (-v,k); reverse=True
    length_map = sorted(length_map.items(), key=lambda kv: (-kv[1], kv[0]), reverse=True)
    if len(length_map) == 0:
        length_map.append((0, 0))
    print(
        cnt,
        mn,
        mx,
        avg,
        ",".join([str(k) for k, _v in length_map]),
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Compute statistics of reads (HISAT2 helper; Python 3 port)."
    )
    parser.add_argument("read_file", nargs="?", type=str, help="reads file")
    parser.add_argument(
        "-n",
        dest="read_count",
        action="store",
        type=int,
        default=10000,
        help="reads count (default: 10000)",
    )
    args = parser.parse_args()

    if not args.read_file:
        parser.print_help()
        sys.exit(1)

    reads_stat(args.read_file, args.read_count)
