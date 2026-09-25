"""List or extract entries from Valve VPK version 1/2 archives."""
import argparse
from dataclasses import dataclass
import fnmatch
from pathlib import Path
import struct


SIGNATURE = 0x55AA1234


@dataclass(frozen=True)
class Entry:
    path: str
    archive_index: int
    offset: int
    length: int
    preload: bytes


class VPK:
    def __init__(self, path):
        self.path = Path(path)
        self.file = self.path.open('rb')
        signature, version, tree_size = struct.unpack('<III', self.file.read(12))
        if signature != SIGNATURE or version not in (1, 2):
            raise ValueError('Unsupported VPK header')
        self.version = version
        self.header_size = 12 if version == 1 else 28
        if version == 2:
            self.file.read(16)
        self.tree_size = tree_size
        self.data_offset = self.header_size + tree_size
        self.entries = self._read_tree()

    def close(self):
        self.file.close()

    def _string(self):
        value = bytearray()
        while True:
            byte = self.file.read(1)
            if not byte:
                raise ValueError('Unexpected end of VPK directory tree')
            if byte == b'\0':
                return value.decode('utf-8')
            value.extend(byte)

    def _read_tree(self):
        entries = []
        tree_end = self.header_size + self.tree_size
        while self.file.tell() < tree_end:
            extension = self._string()
            if not extension:
                break
            while True:
                directory = self._string()
                if not directory:
                    break
                while True:
                    filename = self._string()
                    if not filename:
                        break
                    _, preload_size, archive_index, offset, length, terminator = \
                        struct.unpack('<IHHIIH', self.file.read(18))
                    if terminator != 0xFFFF:
                        raise ValueError('Invalid VPK directory entry terminator')
                    preload = self.file.read(preload_size)
                    directory = '' if directory == ' ' else directory
                    path = f'{directory}/{filename}.{extension}'.lstrip('/')
                    entries.append(Entry(path, archive_index, offset, length, preload))
        return entries

    def read(self, entry):
        if entry.archive_index == 0x7FFF:
            source = self.file
            absolute_offset = self.data_offset + entry.offset
        else:
            stem = self.path.stem
            if stem.endswith('_dir'):
                stem = stem[:-4]
            archive = self.path.with_name(f'{stem}_{entry.archive_index:03d}.vpk')
            source = archive.open('rb')
            absolute_offset = entry.offset
        try:
            source.seek(absolute_offset)
            return entry.preload + source.read(entry.length)
        finally:
            if source is not self.file:
                source.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vpk', type=Path, required=True)
    parser.add_argument('--pattern', default='*')
    parser.add_argument('--extract', help='Exact archive path to extract')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    archive = VPK(args.vpk)
    try:
        if args.extract:
            if args.output is None:
                parser.error('--output is required with --extract')
            matches = [entry for entry in archive.entries
                       if entry.path.lower() == args.extract.lower()]
            if len(matches) != 1:
                raise ValueError(f'Expected one exact entry, found {len(matches)}')
            if args.output.exists():
                raise FileExistsError(f'Refusing to overwrite {args.output}')
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(archive.read(matches[0]))
            print(f'{matches[0].path}\t{args.output}\t{args.output.stat().st_size}')
        else:
            for entry in archive.entries:
                if fnmatch.fnmatch(entry.path.lower(), args.pattern.lower()):
                    print(f'{entry.path}\t{entry.length + len(entry.preload)}')
    finally:
        archive.close()


if __name__ == '__main__':
    main()
