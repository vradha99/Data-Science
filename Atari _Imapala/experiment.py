# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Importance Weighted Actor-Learner Architectures."""

import collections
import contextlib
import functools
import logging
import os
import sys

import numpy as np
import sonnet as snt
import tensorflow as tf
from tqdm import tqdm

import config
import environments
import py_process
import vtrace

FLAGS = config.FLAGS
try:
    import dynamic_batching
except tf.errors.NotFoundError:
    tf.logging.warning("Running without dynamic batching.")

# Set TF CPP level logging level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Set TF Python level logging level
tf.logging.set_verbosity(tf.logging.DEBUG)

root_logger = logging.getLogger()
# create formatter and add it to the handlers
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
#
# # create file handler which logs even debug messages
logfile_handler = logging.FileHandler("experiment.log", mode="w")
# stream_handler = logging.StreamHandler(sys.stderr)
# stream_handler.setLevel(logging.INFO)
# logfile_handler.setLevel(logging.DEBUG)
logfile_handler.setFormatter(formatter)
root_logger.addHandler(logfile_handler)
# logger.addHandler(stream_handler)

nest = tf.contrib.framework.nest

# Structure to be sent from actors to learner.
ActorOutput = collections.namedtuple(
    "ActorOutput", "game_name agent_state env_outputs agent_outputs"
)
AgentOutput = collections.namedtuple(
    "AgentOutput", "action policy_logits baseline")


def is_single_machine():
    return FLAGS.task == -1


class Agent(snt.RNNCore):
    """Agent with ResNet."""

    def __init__(self, num_actions):
        super(Agent, self).__init__(name="agent")

        self._num_actions = num_actions

        with self._enter_variable_scope():
            self._core = tf.contrib.rnn.LSTMBlockCell(256)

    def initial_state(self, batch_size):
        return self._core.zero_state(batch_size, tf.float32)

    def _torso(self, input_):
        last_action, env_output = input_
        reward, _, _, frame = env_output

        # Convert to floats.
        frame = tf.to_float(frame)
        frame = frame

        # frame /= 255
        with tf.variable_scope("convnet"):
            conv_out = frame
            for i, (num_ch, num_blocks) in enumerate(
                    [(16, 2), (32, 2), (32, 2)]):
                # Downscale.
                conv_out = snt.Conv2D(
                    num_ch, 3, stride=1, padding="SAME")(conv_out)
                conv_out = tf.nn.pool(
                    conv_out,
                    window_shape=[3, 3],
                    pooling_type="MAX",
                    padding="SAME",
                    strides=[2, 2],
                )

                # Residual block(s).
                for j in range(num_blocks):
                    with tf.variable_scope("residual_%d_%d" % (i, j)):
                        block_input = conv_out
                        conv_out = tf.nn.relu(conv_out)
                        conv_out = snt.Conv2D(
                            num_ch, 3, stride=1, padding="SAME")(conv_out)
                        conv_out = tf.nn.relu(conv_out)
                        conv_out = snt.Conv2D(
                            num_ch, 3, stride=1, padding="SAME")(conv_out)
                        conv_out += block_input

        conv_out = tf.nn.relu(conv_out)
        conv_out = snt.BatchFlatten()(conv_out)

        conv_out = snt.Linear(256)(conv_out)
        conv_out = tf.nn.relu(conv_out)

        # Append clipped last reward and one hot last action.
        clipped_reward = tf.expand_dims(tf.clip_by_value(reward, -1, 1), -1)
        one_hot_last_action = tf.one_hot(last_action, self._num_actions)
        return tf.concat([conv_out,
                          clipped_reward,
                          one_hot_last_action,
                          ],
                         axis=1)

    def _head(self, core_output):
        policy_logits = snt.Linear(
            self._num_actions,
            name="policy_logits")(core_output)
        baseline = tf.squeeze(
            snt.Linear(
                1,
                name="baseline")(core_output),
            axis=-1)

        # Sample an action from the policy.
        new_action = tf.multinomial(
            policy_logits,
            num_samples=1,
            output_dtype=tf.int32)
        new_action = tf.squeeze(new_action, 1, name="new_action")

        return AgentOutput(new_action, policy_logits, baseline)

    def _build(self, input_, core_state):
        action, env_output = input_
        actions, env_outputs = nest.map_structure(
            lambda t: tf.expand_dims(t, 0), (action, env_output)
        )
        outputs, core_state = self.unroll(actions, env_outputs, core_state)
        return nest.map_structure(
            lambda t: tf.squeeze(
                t, 0), outputs), core_state

    @snt.reuse_variables
    def unroll(self, actions, env_outputs, core_state):
        _, _, done, _ = env_outputs

        torso_outputs = snt.BatchApply(self._torso)((actions, env_outputs))

        # Note, in this implementation we can't use CuDNN RNN to speed things up due
        # to the state reset. This can be XLA-compiled (LSTMBlockCell needs to be
        # changed to implement snt.LSTMCell).
        initial_core_state = self._core.zero_state(
            tf.shape(actions)[1], tf.float32)
        core_output_list = []
        for input_, d in zip(tf.unstack(torso_outputs), tf.unstack(done)):
            # If the episode ended, the core state should be reset before the
            # next.
            core_state = nest.map_structure(
                functools.partial(tf.where, d), initial_core_state, core_state
            )
            core_output, core_state = self._core(input_, core_state)
            core_output_list.append(core_output)

        return snt.BatchApply(
            self._head)(
            tf.stack(core_output_list)), core_state


def build_actor(agent, env, game_name, action_set):
    """Builds the actor loop."""
    # Initial values.
    initial_env_output, initial_env_state = env.initial()
    initial_agent_state = agent.initial_state(1)
    initial_action = tf.zeros([1], dtype=tf.int32)

    dummy_agent_output, _ = agent(
        (initial_action, nest.map_structure(lambda t: tf.expand_dims(t, 0), initial_env_output),)
        , initial_agent_state, )

    initial_agent_output = nest.map_structure(
        lambda t: tf.zeros(t.shape, t.dtype), dummy_agent_output
    )

    # All state that needs to persist across training iterations. This includes
    # the last environment output, agent state and last agent output. These
    # variables should never go on the parameter servers.
    def create_state(t):
        # Creates a unique variable scope to ensure the variable name is
        # unique.
        with tf.variable_scope(None, default_name="state"):
            return tf.get_local_variable(
                t.op.name, initializer=t, use_resource=True)

    persistent_state = nest.map_structure(
        create_state,
        (
            initial_env_state,
            initial_env_output,
            initial_agent_state,
            initial_agent_output,
        ),
    )

    def step(input_, unused_i):
        """Steps through the agent and the environment."""
        env_state, env_output, agent_state, agent_output = input_

        # Run agent.
        action = agent_output[0]
        batched_env_output = nest.map_structure(
            lambda t: tf.expand_dims(t, 0), env_output
        )
        agent_output, agent_state = agent(
            (action, batched_env_output), agent_state)

        # Convert action index to the native action.
        action = agent_output[0][0]
        raw_action = tf.gather(action_set, action)
        env_output, env_state = env.step(raw_action, env_state)

        return env_state, env_output, agent_state, agent_output

    # Run the unroll. `read_value()` is needed to make sure later usage will
    # return the first values and not a new snapshot of the variables.
    first_values = nest.map_structure(
        lambda v: v.read_value(), persistent_state)
    _, first_env_output, first_agent_state, first_agent_output = first_values

    # Use scan to apply `step` multiple times, therefore unrolling the agent
    # and environment interaction for `FLAGS.unroll_length`. `tf.scan` forwards
    # the output of each call of `step` as input of the subsequent call of `step`.
    # The unroll sequence is initialized with the agent and environment states
    # and outputs as stored at the end of the previous unroll.
    # `output` stores lists of all states and outputs stacked along the entire
    # unroll. Note that the initial states and outputs (fed through `initializer`)
    # are not in `output` and will need to be added manually later.
    output = tf.scan(step, tf.range(FLAGS.unroll_length), first_values)
    _, env_outputs, _, agent_outputs = output

    # Update persistent state with the last output from the loop.
    assign_ops = nest.map_structure(
        lambda v, t: v.assign(t[-1]), persistent_state, output
    )

    # The control dependency ensures that the final agent and environment states
    # and outputs are stored in `persistent_state` (to initialize next unroll).
    with tf.control_dependencies(nest.flatten(assign_ops)):
        # Remove the batch dimension from the agent state/output.
        first_agent_state = nest.map_structure(
            lambda t: t[0], first_agent_state)
        first_agent_output = nest.map_structure(
            lambda t: t[0], first_agent_output)
        agent_outputs = nest.map_structure(lambda t: t[:, 0], agent_outputs)

        # Concatenate first output and the unroll along the time dimension.
        full_agent_outputs, full_env_outputs = nest.map_structure(
            lambda first, rest: tf.concat([[first], rest], 0),
            (first_agent_output, first_env_output),
            (agent_outputs, env_outputs),
        )

        output = ActorOutput(
            game_name=game_name,
            agent_state=first_agent_state,
            env_outputs=full_env_outputs,
            agent_outputs=full_agent_outputs,
        )

        # No backpropagation should be done here.
        return nest.map_structure(tf.stop_gradient, output)


def compute_baseline_loss(advantages):
    # Loss for the baseline, summed over the time dimension.
    # Multiply by 0.5 to match the standard update rule:
    # d(loss) / d(baseline) = advantage
    return 0.5 * tf.reduce_sum(tf.square(advantages))


def compute_entropy_loss(logits):
    policy = tf.nn.softmax(logits)
    log_policy = tf.nn.log_softmax(logits)
    entropy_per_timestep = tf.reduce_sum(-policy * log_policy, axis=-1)
    return -tf.reduce_sum(entropy_per_timestep)


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=actions, logits=logits
    )
    advantages = tf.stop_gradient(advantages)
    policy_gradient_loss_per_timestep = cross_entropy * advantages
    return tf.reduce_sum(policy_gradient_loss_per_timestep)


def build_learner(agent, agent_state, env_outputs, agent_outputs):
    """Builds the learner loop.

  Args:
    agent: A snt.RNNCore module outputting `AgentOutput` named tuples, with an
      `unroll` call for computing the outputs for a whole trajectory.
    agent_state: The initial agent state for each sequence in the batch.
    env_outputs: A `StepOutput` namedtuple where each field is of shape
      [T+1, ...].
    agent_outputs: An `AgentOutput` namedtuple where each field is of shape
      [T+1, ...].

  Returns:
    A tuple of (done, infos, and environment frames) where
    the environment frames tensor causes an update.
  """
    learner_outputs, _ = agent.unroll(
        agent_outputs.action, env_outputs, agent_state)

    # Use last baseline value (from the value function) to bootstrap.
    bootstrap_value = learner_outputs.baseline[-1]

    # At this point, the environment outputs at time step `t` are the inputs that
    # lead to the learner_outputs at time step `t`. After the following shifting,
    # the actions in agent_outputs and learner_outputs at time step `t` is what
    # leads to the environment outputs at time step `t`.
    agent_outputs = nest.map_structure(lambda t: t[1:], agent_outputs)
    rewards, infos, done, _ = nest.map_structure(lambda t: t[1:], env_outputs)
    learner_outputs = nest.map_structure(lambda t: t[:-1], learner_outputs)

    if FLAGS.reward_clipping == "abs_one":
        clipped_rewards = tf.clip_by_value(rewards, -1, 1)
    elif FLAGS.reward_clipping == "soft_asymmetric":
        squeezed = tf.tanh(rewards / 5.0)
        # Negative rewards are given less weight than positive rewards.
        clipped_rewards = tf.where(rewards < 0, 0.3 * squeezed, squeezed) * 5.0

    discounts = tf.to_float(~done) * FLAGS.discounting

    # Compute V-trace returns and weights.
    # Note, this is put on the CPU because it's faster than on GPU. It can be
    # improved further with XLA-compilation or with a custom TensorFlow
    # operation.
    with tf.device("/cpu"):
        vtrace_returns = vtrace.from_logits(
            behaviour_policy_logits=agent_outputs.policy_logits,
            target_policy_logits=learner_outputs.policy_logits,
            actions=agent_outputs.action,
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs.baseline,
            bootstrap_value=bootstrap_value,
        )

    # Compute loss as a weighted sum of the baseline loss, the policy gradient
    # loss and an entropy regularization term.
    total_loss = compute_policy_gradient_loss(
        learner_outputs.policy_logits,
        agent_outputs.action,
        vtrace_returns.pg_advantages,
    )
    total_loss += FLAGS.baseline_cost * compute_baseline_loss(
        vtrace_returns.vs - learner_outputs.baseline
    )
    total_loss += FLAGS.entropy_cost * compute_entropy_loss(
        learner_outputs.policy_logits
    )

    # Optimization
    num_env_frames = tf.train.get_global_step()
    learning_rate = tf.train.polynomial_decay(
        FLAGS.learning_rate, num_env_frames, FLAGS.total_environment_frames, 0
    )
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate, FLAGS.decay, FLAGS.momentum, FLAGS.epsilon
    )
    train_op = optimizer.minimize(total_loss)

    # Merge updating the network and environment frames into a single tensor.
    with tf.control_dependencies([train_op]):
        num_env_frames_and_train = num_env_frames.assign_add(
            FLAGS.batch_size * FLAGS.unroll_length * FLAGS.num_action_repeats
        )

    # Adding a few summaries.
    tf.summary.scalar("learning_rate", learning_rate)
    tf.summary.scalar("total_loss", total_loss)
    tf.summary.histogram("action", agent_outputs.action)

    return done, infos, num_env_frames_and_train


def create_environment(game_name, seed, is_test=False):
    """Creates an environment wrapped in a `FlowEnvironment`."""

    # Note, you may want to use a level cache to speed of compilation of
    # environment maps. See the documentation for the Python interface of DeepMind
    # Lab.
    config = {
        "width": FLAGS.width,
        "height": FLAGS.height,
        "logLevel": "WARN",
        "n_input_frames": FLAGS.history,
        "show_display": FLAGS.display,
        "save_video": FLAGS.save_video,
    }
    if is_test:
        config["allowHoldOutLevels"] = "true"
        # Mixer seed for evalution, see
        # https://github.com/deepmind/lab/blob/master/docs/users/python_api.md
        config["mixerSeed"] = 0x600D5EED

    p = py_process.PyProcess(
        environments.PyProcessAtari,
        game_name,
        config,
        FLAGS.num_action_repeats,
        seed,
    )
    return environments.FlowEnvironment(p.proxy)


@contextlib.contextmanager
def pin_global_variables(device):
    """Pins global variables to the specified device."""

    def getter(getter, *args, **kwargs):
        var_collections = kwargs.get("collections", None)
        if var_collections is None:
            var_collections = [tf.GraphKeys.GLOBAL_VARIABLES]
        if tf.GraphKeys.GLOBAL_VARIABLES in var_collections:
            with tf.device(device):
                return getter(*args, **kwargs)
        else:
            return getter(*args, **kwargs)

    with tf.variable_scope("", custom_getter=getter) as vs:
        yield vs


def train(action_set, game_list):
    """Train."""

    if is_single_machine():
        local_job_device = ""
        shared_job_device = ""

        def is_actor_fn(i):
            return True

        is_learner = True
        global_variable_device = "/gpu"
        server = tf.train.Server.create_local_server()
        filters = []
    else:
        local_job_device = "/job:%s/task:%d" % (FLAGS.job_name, FLAGS.task)
        shared_job_device = "/job:learner/task:0"

        def is_actor_fn(
                i):
            return FLAGS.job_name == "actor" and i == FLAGS.task

        is_learner = FLAGS.job_name == "learner"

        # Placing the variable on CPU, makes it cheaper to send it to all the
        # actors. Continual copying the variables from the GPU is slow.
        global_variable_device = shared_job_device + "/cpu"
        cluster = tf.train.ClusterSpec(
            {
                "actor": ["localhost:%d" % (8001 + i) for i in
                          range(FLAGS.num_actors)],
                "learner": ["localhost:8000"],
            }
        )
        server = tf.train.Server(
            cluster, job_name=FLAGS.job_name, task_index=FLAGS.task
        )
        filters = [shared_job_device, local_job_device]

    # Only used to find the actor output structure.
    with tf.Graph().as_default():
        agent = Agent(len(action_set))
        env = create_environment(game_list[0], seed=1)
        structure = build_actor(agent, env, game_list[0], action_set)
        flattened_structure = nest.flatten(structure)
        dtypes = [t.dtype for t in flattened_structure]
        shapes = [t.shape.as_list() for t in flattened_structure]

    with tf.Graph().as_default(), tf.device(
            local_job_device + "/cpu"
    ), pin_global_variables(global_variable_device):
        tf.set_random_seed(FLAGS.seed)  # Makes initialization deterministic.

        # Create Queue and Agent on the learner.
        with tf.device(shared_job_device):
            queue = tf.FIFOQueue(1, dtypes, shapes, shared_name="buffer")
            agent = Agent(len(action_set))

            if is_single_machine() and "dynamic_batching" in sys.modules:
                # For single machine training, we use dynamic batching for improved GPU
                # utilization. The semantics of single machine training are slightly
                # different from the distributed setting because within a single unroll
                # of an environment, the actions may be computed using different weights
                # if an update happens within the unroll.
                old_build = agent._build

                @dynamic_batching.batch_fn
                def build(*args):
                    with tf.device("/gpu"):
                        return old_build(*args)

                tf.logging.info("Using dynamic batching.")
                agent._build = build

        # Build actors and ops to enqueue their output.
        enqueue_ops = []
        for i in range(FLAGS.num_actors):
            if is_actor_fn(i):
                game = game_list[i % len(game_list)]
                tf.logging.info(
                    "Creating actor %d with game environment %s", i, game)
                env = create_environment(game, seed=i + 1)
                actor_output = build_actor(agent, env, game, action_set)
                with tf.device(shared_job_device):
                    enqueue_ops.append(
                        queue.enqueue(
                            nest.flatten(actor_output)))

        # If running in a single machine setup, run actors with QueueRunners
        # (separate threads).
        if is_learner and enqueue_ops:
            tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))

        # Build learner.
        if is_learner:
            # Create global step, which is the number of environment frames
            # processed.
            tf.get_variable(
                "num_environment_frames",
                initializer=tf.zeros_initializer(),
                shape=[],
                dtype=tf.int64,
                trainable=False,
                collections=[
                    tf.GraphKeys.GLOBAL_STEP,
                    tf.GraphKeys.GLOBAL_VARIABLES],
            )

            # Create batch (time major) and recreate structure.
            dequeued = queue.dequeue_many(FLAGS.batch_size)
            dequeued = nest.pack_sequence_as(structure, dequeued)

            def make_time_major(s):
                return nest.map_structure(lambda t: tf.transpose(
                    t, [1, 0] + list(range(t.shape.ndims))[2:]), s, )

            dequeued = dequeued._replace(
                env_outputs=make_time_major(dequeued.env_outputs),
                agent_outputs=make_time_major(dequeued.agent_outputs),
            )

            with tf.device("/gpu"):
                # Using StagingArea allows us to prepare the next batch and send it to
                # the GPU while we're performing a training step. This adds up to 1 step
                # policy lag.
                flattened_output = nest.flatten(dequeued)
                area = tf.contrib.staging.StagingArea(
                    [t.dtype for t in flattened_output],
                    [t.shape for t in flattened_output],
                )
                stage_op = area.put(flattened_output)

                data_from_actors = nest.pack_sequence_as(structure, area.get())

                # Unroll agent on sequence, create losses and update ops.
                output = build_learner(
                    agent,
                    data_from_actors.agent_state,
                    data_from_actors.env_outputs,
                    data_from_actors.agent_outputs,
                )

        # Create MonitoredSession (to run the graph, checkpoint and log).
        tf.logging.info("Creating MonitoredSession, is_chief %s", is_learner)
        config = tf.ConfigProto(
            allow_soft_placement=True,
            device_filters=filters)
        with tf.train.MonitoredTrainingSession(
                server.target,
                is_chief=is_learner,
                checkpoint_dir=FLAGS.logdir,
                save_checkpoint_secs=600,
                save_summaries_secs=30,
                log_step_count_steps=50000,
                config=config,
                hooks=[py_process.PyProcessHook()],
        ) as session:

            if is_learner:
                # Logging.
                game_scores = {game: [] for game in game_list}
                summary_writer = tf.summary.FileWriterCache.get(FLAGS.logdir)

                # Prepare data for first run.
                session.run_step_fn(
                    lambda step_context: step_context.session.run(stage_op)
                )

                # Execute learning and track performance.
                num_env_frames_v = 0
                with tqdm(total=FLAGS.total_environment_frames) as pbar:
                    while num_env_frames_v < FLAGS.total_environment_frames:
                        game_names_v, done_v, infos_v, num_env_frames_v, _ = session.run(
                            (data_from_actors.game_name,) + output + (
                                stage_op,))
                        game_names_v = np.repeat(
                            [game_names_v], done_v.shape[0], 0)

                        for game, episode_return, episode_step in zip(
                                game_names_v[done_v],
                                infos_v.episode_return[done_v],
                                infos_v.episode_step[done_v],
                        ):
                            episode_frames = episode_step * FLAGS.num_action_repeats
                            if isinstance(game, bytes):
                                game = game.decode(
                                    "unicode_escape")

                            tf.logging.debug(
                                "Game: %s Episode return: %f",
                                game,
                                episode_return,
                            )

                            summary = tf.summary.Summary()
                            summary.value.add(
                                tag=game + "/episode_return",
                                simple_value=episode_return,
                            )
                            summary.value.add(
                                tag=game + "/episode_frames",
                                simple_value=episode_frames,
                            )
                            summary_writer.add_summary(
                                summary, num_env_frames_v)
                            # Clear game scores.
                            game_scores = {
                                game: [] for game in game_list
                            }
                            pbar.update(num_env_frames_v - pbar.n)

            else:
                # Execute actors (they just need to enqueue their output).
                while True:
                    session.run(enqueue_ops)


def test(action_set, game_list):
    """Test."""

    game_scores = {game: [] for game in game_list}
    with tf.Graph().as_default():
        agent = Agent(len(action_set))
        outputs = {}
        for game in game_list:
            env = create_environment(game, seed=1, is_test=True)
            outputs[game] = build_actor(
                agent, env, game, action_set)

        with tf.train.SingularMonitoredSession(
                checkpoint_dir=FLAGS.logdir, hooks=[py_process.PyProcessHook()]
        ) as session:
            for game in game_list:
                tf.logging.info("Testing game agent for %s", game)
                while True:
                    done_v, infos_v = session.run(
                        (
                            outputs[game].env_outputs.done,
                            outputs[game].env_outputs.info,
                        )
                    )
                    returns = game_scores[game]
                    returns.extend(infos_v.episode_return[1:][done_v[1:]])

                    if len(returns) >= FLAGS.test_num_episodes:
                        tf.logging.info(
                            "Mean episode return: %f", np.mean(returns))
                        break


def main(_):
    """Main Function"""
    game_list = FLAGS.game
    action_set = environments.COMPLETE_ACTION_SET

    if FLAGS.mode == "train":
        train(action_set, game_list)
    else:
        test(action_set, game_list)


if __name__ == "__main__":
    tf.app.run()

# TODO: Fix logging
# TODO: Check strings- bytes handling in py2 to py3 migration
