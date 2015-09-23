import numpy as np

import mesh
from rapid import ABB_Robot


class RobPath():
    def __init__(self):
        self.mesh = None
        self.rob_parser = ABB_Robot()

    def load_mesh(self, filename):
        self.mesh = mesh.Mesh(filename)
        # TODO: Change bpoints.
        self.mesh.translate(np.float32([20, 20, 0]))
        position = self.mesh.bpoint1  # Rename to position
        size = self.mesh.bpoint2 - self.mesh.bpoint1  # Change by size
        print np.vstack(self.mesh.triangles)

    def translate_mesh(self, position):
        self.mesh.translate(position)

    def resize_mesh(self, size):
        scale = size / self.mesh.size
        self.mesh.scale(scale)

    def set_power(self, power):
        self.rob_parser.power = power

    def set_speed(self, speed):
        self.rob_parser.track_speed = speed

    def set_track(self, height, width, overlap):
        self.track_height = height
        self.track_width = width
        self.track_overlap = overlap
        self.track_distance = (1 - overlap) * width
        print 'Track distance:', self.track_distance
        
    def set_powder(self, carrier_gas, stirrer, turntable):
        self.rob_parser.carrier_gas = carrier_gas
        self.rob_parser.stirrer = stirrer
        self.rob_parser.turntable = turntable

    def init_process(self):
        self.k = 0
        self.path = []
        self.slices = []
        self.pair = False
        self.levels = mesh.get_range_values(self.mesh.z_min,
                                            self.mesh.z_max,
                                            self.track_height)

    def update_process(self):
        slice = self.mesh.get_slice(self.levels[self.k])
        if slice is not None:
            fill_lines = self.mesh.get_grated(slice, self.track_distance)

            # Reverse the order of the slicer fill lines
            if self.pair:
                fill_lines.reverse()
            self.pair = not self.pair

            tool_path = self.mesh.get_path_from_fill_lines(fill_lines)

            self.slices.append(slice)
            self.path.extend(tool_path)
        self.k = self.k + 1
        print 'k, levels:', self.k, len(self.levels)

    def save_rapid(self):
        filename = 'etna.mod'
        directory = 'ETNA'
        routine = self.rob_parser.path2rapid(self.path)
        self.rob_parser.save_file(filename, routine)
        self.rob_parser.upload_file(filename, directory)
        print routine
