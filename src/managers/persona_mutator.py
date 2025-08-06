import argparse
import json
import copy
from pathlib import Path


def load_persona(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def mutate_persona(persona: dict) -> dict:
    """Return a simple mutated version of the given persona."""
    mutated = copy.deepcopy(persona)
    base_id = persona.get('id', 'persona')
    mutated['parent_id'] = base_id
    mutated['id'] = f"{base_id}_m1"
    mutated['generation'] = 1
    mutated['tone'] = 'serious'
    mutated['temperature'] = 0.5
    mutated['expression_style'] = 'reflective and emotionally cautious'
    mutated['response_complexity'] = 'high'
    mutated['preferred_handler'] = 'kimi'
    mutated['values'] = [
        'introspection',
        'emotional safety',
        'loyalty',
        'resilience'
    ]
    mutated['traits'] = {
        'humor': 'dry and intellectual',
        'focus': 'emotional nuance and clarity',
        'touch': 'reserved, protective, affirming'
    }
    return mutated


def main():
    parser = argparse.ArgumentParser(description='Mutate a persona JSON file')
    parser.add_argument('input', help='Path to base persona JSON')
    parser.add_argument('output', help='Path to output mutated persona JSON')
    args = parser.parse_args()

    persona = load_persona(args.input)
    mutated = mutate_persona(persona)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(mutated, f, indent=2, ensure_ascii=False)
    print(f"Mutated persona saved to {out_path}")


if __name__ == '__main__':
    main()