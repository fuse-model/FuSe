"""
Samples final states from succesful trajectories in a validation dataset to use
as goals for evaluation. Logs these goals to an eval_goals.npy file in the same
folder as the dataset. Takes an argument to specify which info key to use as
the success condition.
"""

import jax
import numpy as np
from absl import app, flags, logging
import glob
import os

FLAGS = flags.FLAGS
flags.DEFINE_string("data_path", None, "Location of dataset", required=True)
flags.DEFINE_integer("num_goals", None, "Number of goals", required=True)
flags.DEFINE_string("accept_trajectory_key", None, "Success key", required=True)


def main(_):
    paths = glob.glob(os.path.join(FLAGS.data_path, "val/*.npy"))
    all_traj = [np.load(path, allow_pickle=True) for path in paths]
    all_traj = np.concatenate(all_traj, axis=0)
    success_traj = [traj for traj in all_traj if traj[-1]["observation"][f"info/{FLAGS.accept_trajectory_key}"]]
    success_traj = np.stack(success_traj)

    logging.info(f"Number of successful trajectories: {len(success_traj)}")
    indices = np.random.choice(range(len(success_traj)), size=FLAGS.num_goals, replace=False)
    eval_traj = success_traj[indices]

    eval_initial = jax.tree_map(lambda *xs: np.array(xs), *eval_traj[:, 0])
    eval_goal = jax.tree_map(lambda *xs: np.array(xs), *eval_traj[:, -1])

    eval_goal["observation"]["info/initial_positions"] = eval_initial["observation"]["info/object_positions"]

    np.save(os.path.join(FLAGS.data_path, "eval_goals.npy"), eval_goal)

if __name__ == "__main__":
    app.run(main)
