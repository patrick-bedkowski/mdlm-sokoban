import time

from goal_generating_networks import ConditionalGoalPredictorSokoban, \
    GoalPredictorPixelDiff
from jobs.core import Job
from supervised import DataCreatorSokoban, DataCreatorSokobanPixelDiff
from utils.general_utils import readable_num
import os


class JobTrainSokobanPixelDiff(Job):
    def __init__(self,
                 dataset,
                 dump_folder,
                 steps_into_future,
                 epochs,
                 epochs_checkpoints
                 ):

        # TODO parametrize class choice
        # self.goal_generating_network = ConditionalGoalPredictorSokoban()
        self.goal_generating_network = GoalPredictorPixelDiff()
        self.dataset = dataset
        self.dump_folder = dump_folder
        if not os.path.exists(dump_folder):
            os.makedirs(dump_folder)
        self.steps_into_future = steps_into_future
        self.epochs = epochs
        self.epochs_checkpoints = epochs_checkpoints

        self.data_creator = DataCreatorSokobanPixelDiff()

    def execute(self):
        total_time_start = time.time()

        # 1. Initialize the Transformer
        self.goal_generating_network.construct_networks()

        # 2. Use the JOB's data creator to load and process data
        self.data_creator.load(self.dataset)

        # Receive 3 arrays now
        x_input, y_target, g_goal = self.data_creator.create_xy(self.steps_into_future, 'train')
        vx_input, vy_target, vg_goal = self.data_creator.create_xy(self.steps_into_future, 'validate')

        self.goal_generating_network.fit_and_dump(
            ([x_input, g_goal], y_target),
            ([vx_input, vg_goal], vy_target),
            self.epochs, self.dump_folder,
            checkpoints=self.epochs_checkpoints
        )

        return readable_num(time.time() - total_time_start)