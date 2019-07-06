import tensorflow as tf

flags = tf.app.flags # Creating alias
FLAGS = flags.FLAGS

flags.DEFINE_string("logdir", "tf_log", "TensorFlow log directory.")
flags.DEFINE_enum("mode", "train", ["train", "test"], "Training or test mode.")

# Flags used for testing.
flags.DEFINE_integer("test_num_episodes", 2, "Number of episodes per game.")

# Flags used for distributed training.
flags.DEFINE_integer("task", -1, "Task id. Use -1 for local training.")
flags.DEFINE_enum(
    "job_name",
    "learner",
    ["learner", "actor"],
    "Job name. Ignored when task is set to -1.",
)

# Training.

flags.DEFINE_integer("total_environment_frames", int(
    1e9), "Total environment frames to train for.")
flags.DEFINE_integer("num_actors",15 , "Number of actors.")#15
flags.DEFINE_integer("batch_size", 15, "Batch size for training.")#32
flags.DEFINE_integer("unroll_length", 20, "Unroll length in agent steps.")
flags.DEFINE_integer("num_action_repeats", 4, "Number of action repeats.")
flags.DEFINE_integer("seed", 1, "Random seed.")

# Loss settings.
flags.DEFINE_float("entropy_cost", 0.01, "Entropy cost/multiplier.")
flags.DEFINE_float("baseline_cost", 0.5, "Baseline cost/multiplier.")
flags.DEFINE_float("discounting", 0.99, "Discounting factor.")
flags.DEFINE_enum(
    "reward_clipping", "abs_one", [
        "abs_one", "soft_asymmetric"], "Reward clipping.")

# Environment settings.

flags.DEFINE_list(
    "game",
    "Breakout-v0",
    """Game name or Game names seperated by comma """
    """where games will be assigned round robin to the actors.""",
)
flags.DEFINE_integer("width", 84, "Width of observation.")
flags.DEFINE_integer("height", 84, "Height of observation.")
flags.DEFINE_integer(
    "history", 4, "Number of frames used as input i.e past + current frame"
)
flags.DEFINE_bool("display", False, "Displays the gameplay if set to True")

# Save settings
flags.DEFINE_bool("save_video", False, "Save video of gameplay if set to True")

# Optimizer settings.
flags.DEFINE_float("learning_rate", 0.0006, "Learning rate.")
flags.DEFINE_float("decay", 0.99, "RMSProp optimizer decay.")
flags.DEFINE_float("momentum", 0.0, "RMSProp momentum.")
flags.DEFINE_float("epsilon", 0.01, "RMSProp epsilon.")
