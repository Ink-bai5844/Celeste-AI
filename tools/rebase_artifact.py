"""Write a portable copy of a historical JSON without altering its evidence."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--map', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, destination, map_path = args.source.resolve(), args.output.resolve(), args.map.resolve()
    if source == destination or destination.exists():
        parser.error('Choose a new output file; historical evidence is never overwritten.')
    if not map_path.is_file():
        parser.error('The supplied owned map file does not exist.')
    changes = 0

    def update(value):
        nonlocal changes
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if key in ('map_path', 'map') and isinstance(item, str) and item.lower().endswith('.bin'):
                    result[key] = str(map_path)
                    changes += 1
                elif key == 'path' and isinstance(item, str) and item.replace('\\', '/').split('/')[-1] == map_path.name:
                    result[key] = str(map_path)
                    changes += 1
                else:
                    result[key] = update(item)
            return result
        if isinstance(value, list):
            return [update(item) for item in value]
        return value

    result = update(json.loads(source.read_text(encoding='utf-8-sig')))
    if not changes:
        parser.error('No map reference was found in this JSON.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(destination), 'map_references_rebased': changes}))


if __name__ == '__main__':
    main()
