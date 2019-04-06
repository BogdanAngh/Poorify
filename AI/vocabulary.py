class Vocabulary:
    """
        Class which holds a vocabulary defined with the characters found in the input.
        The indexing starts with 1(0 is reserver for padding)
    """
    def __init__(self, text):
        #build a sit using the characters in the input
        character_set = set("<START>")
        character_set.update(text)

        #build a characacter => unique index map
        self.char_to_idx = {chr:(idx+1) for (idx, chr) in enumerate(character_set)}
        #build a unique index => character map
        self.idx_to_char = {(idx+1):chr for (idx, chr) in enumerate(character_set)}
        
    def size(self):
        return len(self.char_to_idx)

    def __str__(self):
        return str(self.char_to_idx)