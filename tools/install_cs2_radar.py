"""Install the fixed visible-radar config into an explicitly selected CS2 cfg folder."""
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'cs2_bridge' / 'flywire_radar.cfg'


def install(cfg_directory):
    cfg_directory = Path(cfg_directory).resolve()
    if cfg_directory.name.lower() != 'cfg' or cfg_directory.parent.name.lower() != 'csgo':
        raise ValueError('Expected the game/csgo/cfg directory from a CS2 installation')
    if not cfg_directory.is_dir():
        raise FileNotFoundError(cfg_directory)
    destination = cfg_directory / SOURCE.name
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    destination.write_text(SOURCE.read_text())
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg-directory', type=Path, required=True)
    args = parser.parse_args()
    print(f'Installed {install(args.cfg_directory)}')


if __name__ == '__main__':
    main()
