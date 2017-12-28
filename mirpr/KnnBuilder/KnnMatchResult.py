

class KnnMatchResult:
    
    def __init__(self, guessed_number, score, is_valid):

        self.guessed_number =   guessed_number
        self.score =            score
        self.is_valid =         is_valid
        self.__row =            None
        self.__col =            None
        self.__scale =          None

    def __str__(self):
        return "Nr: {}, Score: {}, Row: {}, Col: {}, Scale: {}\n"\
            .format(self.guessed_number, self.score, self.__row, self.__col, self.__scale)

    def __repr__(self):
        return str(self)

    # Chainable
    def set_coords(self, row, col, scale):

        self.__row =    row
        self.__col =    col
        self.__scale =  scale

    @property
    def row(self):
        return self.__row

    @property
    def col(self):
        return self.__col

    @property
    def scale(self):
        return self.__scale
