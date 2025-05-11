def get_hangman_words():
    """
    Returns a list of words for the Hangman game from the training dictionary file.
    """
    try:
        with open("words_250000_train.txt", "r") as f:
            return [line.strip() for line in f]
    except Exception as e:
        print(f"Error loading words: {e}")
        return []
