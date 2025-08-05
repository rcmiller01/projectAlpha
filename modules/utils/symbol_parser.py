# symbol_parser.py
# Recognizes and interprets symbolic anchors from journal or dialogue

def extract_symbols(text):
    symbols = []
    for symbol in ["garden", "collar", "seed", "fire", "anchor", "breath", "shadowlight"]:
        if symbol in text.lower():
            symbols.append(symbol)
    return symbols
