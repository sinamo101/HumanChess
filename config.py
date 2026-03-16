# Shared model hyperparameters
# Change these values here and they will propagate to all scripts.

ACTION_SIZE = 31       # vocab size for FEN tokens
SEQ_LEN = 77           # FEN sequence length
D_MODEL = 256          # transformer embedding dimension
NUM_LAYERS = 8         # number of transformer blocks
NUM_HEADS = 8          # number of attention heads
D_FF = D_MODEL * 4     # feedforward hidden dimension
DROPOUT = 0.1
OUTPUT_SIZE = 128      # win% bucket count for base model
MAX_DISTANCE = 8       # relative position bias range (8x8 board)
