CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']

CTLABELS_37 = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9',]

from unidecode import unidecode

ignored_default = ['°']

ignored_37 = ignored_default + [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/',':',';','<','=','>','?','@','[','\\',']','^','_','`','{','|','}','~']

import bs4

def parse_html_tags(text):
    soup = bs4.BeautifulSoup(text, 'html.parser')
    return soup.get_text()


def decode_text_96(text):
    return ''.join([CTLABELS[c] for c in text if c < 95])

def decode_text_37(text):
    return ''.join([CTLABELS_37[c] for c in text if c < 36])

def encode_text_96(text, align_to=25, strict = False, upper = False, remove_accents = True, ignore = []):
    text = parse_html_tags(text)

    for c in ignore:
        text = text.replace(c, '')

    for c in ignored_default:
        text = text.replace(c, '')

    #if remove_accents:
    #    text = unidecode(text)

    if upper:
        text = text.upper()

    if not strict:
        encoded = [CTLABELS.index(c) if c in CTLABELS else 95 for c in text]
        unknowns = [c for c in text if c not in CTLABELS]
        if len(unknowns) > 0:
            print(f"WARNING: Unknown characters: {unknowns}")
    else:
        encoded = [CTLABELS.index(c) for c in text]

    if align_to is None:
        return encoded

    if isinstance(align_to, int):
        if align_to > len(encoded):
            encoded += [96] * (align_to - len(encoded))
        else:
            print("WARNING: Align to is less than the length of the encoded text")
            return None
    else:
        print("WARNING: Align to is not an integer")
        return None
    
    return encoded

def encode_text_37(text, align_to=25, strict = False, remove_accents = True, ignore = []):
    text = parse_html_tags(text)

    for c in ignore:
        text = text.replace(c, '')

    for c in ignored_37:
        text = text.replace(c, '')

    #if remove_accents:
    #    text = unidecode(text)

    text = text.upper()

    if not strict:
        encoded = [CTLABELS_37.index(c) if c in CTLABELS_37 else 36 for c in text]
        unknowns = [c for c in text if c not in CTLABELS_37]
        if len(unknowns) > 0:
            print(f"WARNING: Unknown characters in text: {unknowns}")
    else:
        encoded = [CTLABELS_37.index(c) for c in text]

    if align_to is None:
        return encoded

    if isinstance(align_to, int):
        if align_to > len(encoded):
            encoded += [37] * (align_to - len(encoded))
        else:
            print("WARNING: Align to is less than the length of the encoded text")
            return None
    else:
        print("WARNING: Align to is not an integer")
        return None
    
    return encoded