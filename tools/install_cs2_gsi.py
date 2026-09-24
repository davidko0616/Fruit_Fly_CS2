"""Install the read-only FlyWire GSI config into an explicitly selected CS2 cfg folder."""
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'cs2_bridge' / 'gamestate_integration_flywire.cfg'


def install(cfg_directory, token):
    cfg_directory = Path(cfg_directory).resolve()
    if cfg_directory.name.lower() != 'cfg' or cfg_directory.parent.name.lower() != 'csgo':
        raise ValueError('Expected the game/csgo/cfg directory from a CS2 installation')
    if not cfg_directory.is_dir():
        raise FileNotFoundError(cfg_directory)
    if not token or token == 'replace-with-a-local-token' or any(ch.isspace() for ch in token):
        raise ValueError('Supply a non-placeholder token without whitespace')
    destination = cfg_directory / SOURCE.name
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    text = SOURCE.read_text().replace('replace-with-a-local-token', token)
    destination.write_text(text)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg-directory', type=Path, required=True)
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    print(f'Installed {install(args.cfg_directory, args.token)}')


if __name__ == '__main__':
    main()
