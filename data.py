import torch as tc
import itertools

def get_all_sequences(length, vocab_size):
    """
    Generate all possible sequences of a given length and vocabulary size 
    """
    sequences = itertools.product(range(vocab_size), repeat=length)
    sequences = [tc.tensor(seq) for seq in sequences]
    return sequences

def get_train_test_split(sequences, train_test_split):
    """
    Split the sequences into train and test sets
    """
    train_size = int(len(sequences) * train_test_split)
    train_sequences = sequences[:train_size]
    test_sequences = sequences[train_size:]
    return train_sequences, test_sequences

if __name__ == "__main__":
    input_size = 3
    vocab_size = 2

    sequences = get_all_sequences(input_size, vocab_size)
    print(f"generated {len(sequences)} sequences of length {input_size} with vocabulary size {vocab_size}")

    train_test_split = 0.7
    train_sequences, test_sequences = get_train_test_split(sequences, train_test_split)
    print(f"train size: {len(train_sequences)}")
    print(f"test size: {len(test_sequences)}")
