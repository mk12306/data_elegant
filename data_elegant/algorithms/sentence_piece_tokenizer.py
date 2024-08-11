import sentencepiece


class SentencePieceTokenizer(object):
    def __init__(self, path_sentence_piece_model='./algorithms/sentence_piece/zh.sp.model'):
        self.path_sentence_piece_model = path_sentence_piece_model
        self.sentence_piece_model = None

    def init_model(self):
        if self.sentence_piece_model is None:
            self.sentence_piece_model = sentencepiece.SentencePieceProcessor()
            self.sentence_piece_model.load(self.path_sentence_piece_model)

    def tokenize(self, content, join_on_whitespace=False):
        self.init_model()
        content_tokenized = self.sentence_piece_model.encode_as_pieces(content)

        if join_on_whitespace:
            content_tokenized = ' '.join(content_tokenized)
        return content_tokenized

    def __reduce__(self):
        return (
            self.__class__,
            (self.path_sentence_piece_model,),
        )
