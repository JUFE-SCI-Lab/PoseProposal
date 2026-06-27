from mmcv.runner.hooks import Hook
from mmcv.runner.dist_utils import get_dist_info
from vkits.vbasic import draw_curves
import os


class DrawHook(Hook):
    def __init__(self, log_file_path, save_path, draw_interval=10):
        # draw_curves(f'output/20240330_101414.log.json', 'output/curve.png')
        self.log_file_path = log_file_path
        self.save_path = save_path
        self.draw_interval = draw_interval
        return

    def after_epoch(self, runner):
        runner.logger.info(f'In draw hook rank {get_dist_info()[0]}...')

        if get_dist_info()[0] == 0:
            if (runner.epoch) % self.draw_interval == 0:
                self._do_draw(runner.logger.info)
        else:
            return

    def _do_draw(self, printf=print):
        printf(f'Drawing {self.log_file_path} by rank {get_dist_info()[0]}...')
        try:
            for filename in os.listdir(os.path.dirname(self.log_file_path)):
                if filename.endswith('best.png'):
                    prev_best_path = f'{os.path.dirname(self.log_file_path)}/{filename}'
                    printf(f'Removing {prev_best_path} by rank {get_dist_info()[0]}...')
                    os.remove(prev_best_path)
            draw_curves(self.log_file_path, self.save_path)
        except Exception as e:
            printf(f'{e}')
            printf(f'Fail to draw curves...')

    def after_run(self, runner):
        self._do_draw()
