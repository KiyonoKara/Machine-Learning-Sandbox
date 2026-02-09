from collections import Counter
import numpy as np

# constants
SENTENCE_BEGIN = "<s>"
SENTENCE_END = "</s>"
# unknown token
UNK = "<UNK>"


def create_ngrams(tokens: list, n: int) -> list:
    """Creates n-grams for the given token sequence.
  Args:
    tokens (list): a list of tokens as strings
    n (int): the length of n-grams to create

  Returns:
    list: list of tuples of strings, each tuple being one of the individual n-grams
  """
    grams = []
    for i in range(len(tokens) - n + 1):
        grams.append(tuple(tokens[i:i + n]))
    return grams


def read_file(path: str) -> list:
    """
  Reads the contents of a file in line by line.
  Args:
    path (str): the location of the file to read

  Returns:
    list: list of strings, the contents of the file
  """
    f = open(path, "r", encoding="utf-8")
    contents = f.readlines()
    f.close()
    return contents


def tokenize_line(line: str, ngram: int,
                  by_char: bool = True,
                  sentence_begin: str = SENTENCE_BEGIN,
                  sentence_end: str = SENTENCE_END):
    """
  Tokenize a single string. Glue on the appropriate number of
  sentence begin tokens and sentence end tokens (ngram - 1), except
  for the case when ngram == 1, when there will be one sentence begin
  and one sentence end token.
  Args:
    line (str): text to tokenize
    ngram (int): ngram preparation number
    by_char (bool): default value True, if True, tokenize by character, if
      False, tokenize by whitespace
    sentence_begin (str): sentence begin token value
    sentence_end (str): sentence end token value

  Returns:
    list of strings - a single line tokenized
  """
    inner_pieces = None
    if by_char:
        inner_pieces = list(line)
    else:
        # otherwise split on white space
        inner_pieces = line.split()

    if ngram == 1:
        tokens = [sentence_begin] + inner_pieces + [sentence_end]
    else:
        tokens = ([sentence_begin] * (ngram - 1)) + inner_pieces + ([sentence_end] * (ngram - 1))
    # always count the unigrams
    return tokens


def tokenize(data: list, ngram: int,
             by_char: bool = True,
             sentence_begin: str = SENTENCE_BEGIN,
             sentence_end: str = SENTENCE_END):
    """
  Tokenize each line in a list of strings. Glue on the appropriate number of
  sentence begin tokens and sentence end tokens (ngram - 1), except
  for the case when ngram == 1, when there will be one sentence begin
  and one sentence end token.
  Args:
    data (list): list of strings to tokenize
    ngram (int): ngram preparation number
    by_char (bool): default value True, if True, tokenize by character, if
      False, tokenize by whitespace
    sentence_begin (str): sentence begin token value
    sentence_end (str): sentence end token value

  Returns:
    list of strings - all lines tokenized as one large list
  """
    total = []
    # also glue on sentence begin and end items
    for line in data:
        line = line.strip()
        # skip empty lines
        if len(line) == 0:
            continue
        tokens = tokenize_line(line, ngram, by_char, sentence_begin, sentence_end)
        total += tokens
    return total


class LanguageModel:

    def __init__(self, n_gram):
        """Initializes an untrained LanguageModel
    Args:
      n_gram (int): the n-gram order of the language model to create
    """
        # N (n)
        self.n_gram: int = n_gram
        # N gram counts
        self.n_gram_counts: Counter = Counter()
        # V
        self.vocabulary: set[str] = set()
        # |V|
        self.vocabulary_size: int = 0
        # |tokens|
        self.total_tokens: int = 0
        # n - 1 gram counts
        self.context_counts: Counter = Counter()

    def train(self, tokens: list, verbose: bool = False) -> None:
        """Trains the language model on the given data. Assumes that the given data
    has tokens that are white-space separated, has one sentence per line, and
    that the sentences begin with <s> and end with </s>
    Args:
      tokens (list): tokenized data to be trained on as a single list
      verbose (bool): default value False, to be used to turn on/off debugging prints
    """
        # Count frequency of each token
        token_counts = Counter(tokens)

        # Replace infrequent tokens (that appear one time) with UNK, this is only added once
        filtered_tokens = [token if token_counts[token] > 1 else UNK for token in tokens]

        # Create n-grams and (n - 1) grams and then count their frequencies
        n_grams = create_ngrams(filtered_tokens, self.n_gram)
        self.n_gram_counts = Counter(n_grams)

        # Set the class' vocabulary and its cardinality
        self.vocabulary = set(filtered_tokens)
        self.vocabulary_size = len(self.vocabulary)

        # Account for unigram cases
        if self.n_gram == 1:
            self.total_tokens = len(filtered_tokens)
        else:
            # Account for n - 1 gram cases
            self.context_counts = Counter(create_ngrams(filtered_tokens, self.n_gram - 1))

        # Verbose for debugging
        if verbose:
            print("Total tokens:", len(filtered_tokens))
            print("Vocabulary size:", self.vocabulary_size)
            print("N-gram count size:", len(self.n_gram_counts))

    def score(self, sentence_tokens: list) -> float:
        """Calculates the probability score for a given string representing a single sequence of tokens.
    Args:
      sentence_tokens (list): a tokenized sequence to be scored by this model

    Returns:
      float: the probability value of the given tokens for this model
    """
        # Add from vocabulary or unknown
        processed_sentence_tokens = []
        for token in sentence_tokens:
            if token in self.vocabulary:
                processed_sentence_tokens.append(token)
            else:
                processed_sentence_tokens.append(UNK)

        # Set sentence tokens back to the processed tokens
        sentence_tokens = processed_sentence_tokens

        log_probability = 0
        # Unigram case
        if self.n_gram == 1:
            for token in sentence_tokens:
                unigram_to_score = tuple([token])
                # Reference
                # P(w_i) = \frac{count(w_i) + 1}{N + |V|}
                unigram_probability = (
                        (self.n_gram_counts[unigram_to_score] + 1) / (self.total_tokens + self.vocabulary_size))
                log_probability += np.log(unigram_probability)

            return np.exp(log_probability)
        # Bi-gram or greater case
        else:
            n = self.n_gram
            for i in range(len(sentence_tokens) - n + 1):
                n_gram_scoring_target = tuple(sentence_tokens[i:i + n], )
                prefix_target = tuple(sentence_tokens[i:i + n - 1], )
                # Reference
                # P(w_i | w_{i - N + 1}...w_{i - 1})
                # = \frac{count(w_{i - N + 1}...w_i) + 1}{count(w_{i - N + 1}...w_{i - 1}) + |V|}
                n_gram_probability = ((self.n_gram_counts[n_gram_scoring_target] + 1)
                                      / (self.context_counts[prefix_target] + self.vocabulary_size))
                log_probability += np.log(n_gram_probability)

            return np.exp(log_probability)

    def generate_sentence(self) -> list:
        """Generates a single sentence from a trained language model using the Shannon technique.

    Returns:
      list: the generated sentence as a list of tokens
    """
        # Unigram case
        if self.n_gram == 1:
            # SENTENCE_BEGIN constant
            tokens_uni = [SENTENCE_BEGIN]

            # Get tokens by their counts in the training data then use them as weights
            items = [(unigram[0], count)
                     for (unigram, count) in self.n_gram_counts.items() if unigram[0] != SENTENCE_BEGIN]
            # Choices
            choices: list[str] = [unigram_token for (unigram_token, _) in items]
            # Weights
            weights: list[int] = [count for (_, count) in items]

            # Turn counts into probabilities by dividing them by the total count
            weights = [count / np.sum(weights) for count in weights]

            # Get random tokens until SENTENCE_END is reached
            while tokens_uni[-1] != SENTENCE_END:
                next_token: str = np.random.choice(choices, p=weights)
                tokens_uni.append(next_token)
            return tokens_uni
        # Bi-gram or greater case
        else:
            # SENTENCE_BEGIN constant for n - 1
            tokens_n = ([SENTENCE_BEGIN] * (self.n_gram - 1))

            # End sentence when the last n - 1 token is SENTENCE_END
            while tokens_n[-1 * (self.n_gram - 1):] != ([SENTENCE_END] * (self.n_gram - 1)):
                # Prefix for the next generated n - 1 token
                prefix = tokens_n[-1 * (self.n_gram - 1):]

                # Find all n-grams where first n - 1 tokens match the prefix
                # Get last element of the n-gram
                items = [(ngram[-1], count) for (ngram, count) in self.n_gram_counts.items() if
                         list(ngram[:-1]) == prefix and ngram[-1] != SENTENCE_BEGIN]
                # Choices
                choices: list[str] = [token for (token, _) in items]
                # Weights
                weights: list[int] = [count for (_, count) in items]

                # Turn counts into probabilities by dividing them by the total count
                weights = [count / np.sum(weights) for count in weights]

                # Choose the next token using the weights from the probabilities
                next_token: str = np.random.choice(choices, p=weights)
                tokens_n.append(next_token)
            return tokens_n

    def generate(self, n: int) -> list:
        """Generates n sentences from a trained language model using the Shannon technique.
    Args:
      n (int): the number of sentences to generate

    Returns:
      list: a list containing lists of strings, one per generated sentence
    """
        return [self.generate_sentence() for _ in range(n)]

    def perplexity(self, sequence: list) -> float:
        """Calculates the perplexity score for a given sequence of tokens.
    Args:
      sequence (list): a tokenized sequence to be evaluated for perplexity by this model

    Returns:
      float: the perplexity value of the given sequence for this model
    """
        # Get the score of the sequence
        score = self.score(sequence)
        # Count tokens that aren't the SENTENCE_BEGIN constant
        token_count = len([s for s in sequence if s != SENTENCE_BEGIN])

        # Reference
        # perplexity = score^{\frac{-1}{count(tokens)}}
        perplexity = score ** (-1 / token_count)
        return perplexity
