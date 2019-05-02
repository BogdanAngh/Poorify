class Vocabulary:
    """
        Class which holds a vocabulary defined with the words found in the input.
        The indexing starts with 1(0 is reserver for padding)
    """
    def __init__(self, text):
        #build a set using the words in the input
        words_set = set("<START>")
        words_set.update(text.split())

        #build a word => unique index map
        self.word_to_idx = {word:(idx+1) for (idx, word) in enumerate(words_set)}
        #build a unique index => word map
        self.idx_to_word = {(idx+1):word for (idx, word) in enumerate(words_set)}
        
    def size(self):
        return len(self.word_to_idx)

    def __str__(self):
        return str(self.word_to_idx)