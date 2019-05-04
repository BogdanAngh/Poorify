class Vocabulary:
    """
        Class which holds a vocabulary defined with the words found in the input.
        The indexing starts with 1(0 is reserver for padding)
    """
    def __init__(self, text=None):
        if text is not None:
            self.update(text)
        
    def update(self, text):
        words_set = set("<START>")
        words_set.update(text.split())

        #build a word => unique index map
        self.word_to_idx = {word:(idx+1) for (idx, word) in enumerate(words_set)}
        #build a unique index => word map
        self.idx_to_word = {(idx+1):word for (idx, word) in enumerate(words_set)}

    def add_unkw(self):
        vocab_size = self.size()

        self.idx_to_word[vocab_size] = 'UNKW'
        self.word_to_idx['UNKW'] = vocab_size

    def size(self):
        return len(self.word_to_idx)

    def __str__(self):
        return str(self.word_to_idx)