def mk_to_sl():
    return {
        "a": "а",
        "b": "б",
        "v": "в",
        "g": "г",
        "d": "д",
        "gj": "ѓ",
        "e": "е",
        "ž": "ж",
        "z": "з",
        "dz": "ѕ",
        "i": "и",
        "j": "ј",
        "k": "к",
        "l": "л",
        "lj": "љ",
        "m": "м",
        "n": "н",
        "nj": "њ",
        "o": "о",
        "p": "п",
        "r": "р",
        "s": "с",
        "t": "т",
        "kj": "ќ",
        "u": "у",
        "f": "ф",
        "h": "х",
        "c": "ц",
        "č": "ч",
        "dž": "џ",
        "š": "ш"
    }

def transliterate(word: str):
    table = mk_to_sl()
    result = ""
    for i in range(len(word)):
        if i < len(word) - 1 and word[i:i+1] in table:
            result += table[word[i:i+1]]
            i += 1
        elif word[i] in table:
            result += table[word[i]]
        else:
            print(f"[ERROR] Letter {word[i]} not found in the transliteration table!")
    return result
