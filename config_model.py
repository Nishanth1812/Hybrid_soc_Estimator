
MODEL_CONFIG = {

    "input_features": 3,
    "sequence_length": 100,

    "lstm_hidden_1": 64,
    "lstm_hidden_2": 32,

    "dense_units": 16,

    "dropout": 0.2
}

VALIDATION_CONFIG = {

    "soc_min": 0.0,
    "soc_max": 1.0,

    "expected_features": 3,
    "expected_seq_len": 100
}