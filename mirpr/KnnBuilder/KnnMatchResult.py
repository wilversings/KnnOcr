



class KnnMatchResult:
    
    def __init__(self, guessed_number, score, is_valid):
        self.guessed_number = guessed_number
        self.score = score
        self.is_valid = is_valid

    def __str__(self):
        return "Nr: {}, Score: {}\n".format(self.guessed_number, self.score)

    def __repr__(self):
        return str(self)