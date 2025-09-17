import numpy as np
import pandas as pd
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class UAVTrack112Dataset(BaseDataset):
    """ UAVTrack112 dataset.
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.uavtrack112_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path, 
        sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext) for frame_num in range(start_frame+init_omit, end_frame+1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        try:
            # ground_truth_rect = np.genfromtxt(anno_path, delimiter=',', dtype=np.float64)
            df = pd.read_csv(anno_path, header=None, sep=',| ', engine='python')
            ground_truth_rect = df.to_numpy(dtype=np.float64)
        except Exception as e:
            print(f'anno:{anno_path}, error:{e}')
            
        return Sequence(sequence_info['name'], frames, 'uavtrack112', ground_truth_rect[init_omit:,:],
                        object_class=sequence_info['object_class'])

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [
            { 'name':'air conditioning box1', 'path':'data_seq/air conditioning box1', 'startFrame': 1, 'endFrame':437, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/air conditioning box1.txt', 'object_class': 'unknown' },
            { 'name':'air conditioning box2', 'path':'data_seq/air conditioning box2', 'startFrame': 1, 'endFrame':1847, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/air conditioning box2.txt', 'object_class': 'unknown' },
            { 'name':'basketball player1', 'path':'data_seq/basketball player1', 'startFrame': 1, 'endFrame':1047, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/basketball player1.txt', 'object_class': 'unknown' },
            { 'name':'basketball player1_1-n', 'path':'data_seq/basketball player1_1-n', 'startFrame': 1, 'endFrame':312, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/basketball player1_1-n.txt', 'object_class': 'unknown' },
            { 'name':'basketball player1_2-n', 'path':'data_seq/basketball player1_2-n', 'startFrame': 1, 'endFrame':1567, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/basketball player1_2-n.txt', 'object_class': 'unknown' },
            { 'name':'basketball player2-n', 'path':'data_seq/basketball player2-n', 'startFrame': 1, 'endFrame':1225, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/basketball player2-n.txt', 'object_class': 'unknown' },
            { 'name':'basketball player2', 'path':'data_seq/basketball player2', 'startFrame': 1, 'endFrame':1302, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/basketball player2.txt', 'object_class': 'unknown' },
            { 'name':'basketball player3-n', 'path':'data_seq/basketball player3-n', 'startFrame': 1, 'endFrame':732, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/basketball player3-n.txt', 'object_class': 'unknown' },
            { 'name':'basketball player3', 'path':'data_seq/basketball player3', 'startFrame': 1, 'endFrame':1320, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/basketball player3.txt', 'object_class': 'unknown' },
            { 'name':'basketball player4-n', 'path':'data_seq/basketball player4-n', 'startFrame': 1, 'endFrame':408, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/basketball player4-n.txt', 'object_class': 'unknown' },
            { 'name':'bell tower', 'path':'data_seq/bell tower', 'startFrame': 1, 'endFrame':2061, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/bell tower.txt', 'object_class': 'unknown' },
            { 'name':'bike1', 'path':'data_seq/bike1', 'startFrame': 1, 'endFrame':180, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/bike1.txt', 'object_class': 'unknown' },
            { 'name':'bike2', 'path':'data_seq/bike2', 'startFrame': 1, 'endFrame':840, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/bike2.txt', 'object_class': 'unknown' },
            { 'name':'bike3', 'path':'data_seq/bike3', 'startFrame': 1, 'endFrame':234, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/bike3.txt', 'object_class': 'unknown' },
            { 'name':'bike4_1', 'path':'data_seq/bike4_1', 'startFrame': 1, 'endFrame':686, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/bike4_1.txt', 'object_class': 'unknown' },
            { 'name':'bike4_2', 'path':'data_seq/bike4_2', 'startFrame': 1, 'endFrame':1129, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/bike4_2.txt', 'object_class': 'unknown' },
            { 'name':'bike5', 'path':'data_seq/bike5', 'startFrame': 1, 'endFrame':1188, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/bike5.txt', 'object_class': 'unknown' },
            { 'name':'bike6', 'path':'data_seq/bike6', 'startFrame': 1, 'endFrame':1118, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/bike6.txt', 'object_class': 'unknown' },
            { 'name':'bike7_1', 'path':'data_seq/bike7_1', 'startFrame': 1, 'endFrame':301, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/bike7_1.txt', 'object_class': 'unknown' },
            { 'name':'bike7_2', 'path':'data_seq/bike7_2', 'startFrame': 1, 'endFrame':338, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/bike7_2.txt', 'object_class': 'unknown' },
            { 'name':'bike8', 'path':'data_seq/bike8', 'startFrame': 1, 'endFrame':327, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/bike8.txt', 'object_class': 'unknown' },
            { 'name':'bike9_1', 'path':'data_seq/bike9_1', 'startFrame': 1, 'endFrame':401, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/bike9_1.txt', 'object_class': 'unknown' },
            { 'name':'bike9_2', 'path':'data_seq/bike9_2', 'startFrame': 1, 'endFrame':321, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/bike9_2.txt', 'object_class': 'unknown' },
            { 'name':'building1_1', 'path':'data_seq/building1_1', 'startFrame': 1, 'endFrame':1140, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/building1_1.txt', 'object_class': 'unknown' },
            { 'name':'building1_2', 'path':'data_seq/building1_2', 'startFrame': 1, 'endFrame':1036, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/building1_2.txt', 'object_class': 'unknown' },
            { 'name':'bus1-n', 'path':'data_seq/bus1-n', 'startFrame': 1, 'endFrame':1092, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/bus1-n.txt', 'object_class': 'unknown' },
            { 'name':'bus2-n', 'path':'data_seq/bus2-n', 'startFrame': 1, 'endFrame':716, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/bus2-n.txt', 'object_class': 'unknown' },
            { 'name':'car1-n', 'path':'data_seq/car1-n', 'startFrame': 1, 'endFrame':1291, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/car1-n.txt', 'object_class': 'unknown' },
            { 'name':'car1', 'path':'data_seq/car1', 'startFrame': 1, 'endFrame':1509, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/car1.txt', 'object_class': 'unknown' },
            { 'name':'car10', 'path':'data_seq/car10', 'startFrame': 1, 'endFrame':291, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/car10.txt', 'object_class': 'unknown' },
            { 'name':'car11', 'path':'data_seq/car11', 'startFrame': 1, 'endFrame':142, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/car11.txt', 'object_class': 'unknown' },
            { 'name':'car12', 'path':'data_seq/car12', 'startFrame': 1, 'endFrame':78, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/car12.txt', 'object_class': 'unknown' },
            { 'name':'car13', 'path':'data_seq/car13', 'startFrame': 1, 'endFrame':184, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/car13.txt', 'object_class': 'unknown' },
            { 'name':'car14', 'path':'data_seq/car14', 'startFrame': 1, 'endFrame':1690, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/car14.txt', 'object_class': 'unknown' },
            { 'name':'car15', 'path':'data_seq/car15', 'startFrame': 1, 'endFrame':159, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/car15.txt', 'object_class': 'unknown' },
            { 'name':'car16_1', 'path':'data_seq/car16_1', 'startFrame': 1, 'endFrame':100, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/car16_1.txt', 'object_class': 'unknown' },
            { 'name':'car16_2', 'path':'data_seq/car16_2', 'startFrame': 1, 'endFrame':254, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/car16_2.txt', 'object_class': 'unknown' },
            { 'name':'car16_3', 'path':'data_seq/car16_3', 'startFrame': 1, 'endFrame':168, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/car16_3.txt', 'object_class': 'unknown' },
            { 'name':'car17', 'path':'data_seq/car17', 'startFrame': 1, 'endFrame':221, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/car17.txt', 'object_class': 'unknown' },
            { 'name':'car18', 'path':'data_seq/car18', 'startFrame': 1, 'endFrame':171, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/car18.txt', 'object_class': 'unknown' },
            { 'name':'car2-n', 'path':'data_seq/car2-n', 'startFrame': 1, 'endFrame':489, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/car2-n.txt', 'object_class': 'unknown' },
            { 'name':'car2', 'path':'data_seq/car2', 'startFrame': 1, 'endFrame':486, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/car2.txt', 'object_class': 'unknown' },
            { 'name':'car3', 'path':'data_seq/car3', 'startFrame': 1, 'endFrame':2371, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/car3.txt', 'object_class': 'unknown' },
            { 'name':'car4', 'path':'data_seq/car4', 'startFrame': 1, 'endFrame':1302, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/car4.txt', 'object_class': 'unknown' },
            { 'name':'car5', 'path':'data_seq/car5', 'startFrame': 1, 'endFrame':2625, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/car5.txt', 'object_class': 'unknown' },
            { 'name':'car6_1', 'path':'data_seq/car6_1', 'startFrame': 1, 'endFrame':1920, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/car6_1.txt', 'object_class': 'unknown' },
            { 'name':'car6_2', 'path':'data_seq/car6_2', 'startFrame': 1, 'endFrame':1123, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/car6_2.txt', 'object_class': 'unknown' },
            { 'name':'car7_1', 'path':'data_seq/car7_1', 'startFrame': 1, 'endFrame':973, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/car7_1.txt', 'object_class': 'unknown' },
            { 'name':'car7_2', 'path':'data_seq/car7_2', 'startFrame': 1, 'endFrame':1968, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/car7_2.txt', 'object_class': 'unknown' },
            { 'name':'car7_3', 'path':'data_seq/car7_3', 'startFrame': 1, 'endFrame':1596, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/car7_3.txt', 'object_class': 'unknown' },
            { 'name':'car8', 'path':'data_seq/car8', 'startFrame': 1, 'endFrame':183, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/car8.txt', 'object_class': 'unknown' },
            { 'name':'car9_1', 'path':'data_seq/car9_1', 'startFrame': 1, 'endFrame':1745, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/car9_1.txt', 'object_class': 'unknown' },
            { 'name':'car9_2', 'path':'data_seq/car9_2', 'startFrame': 1, 'endFrame':1885, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/car9_2.txt', 'object_class': 'unknown' },
            { 'name':'container', 'path':'data_seq/container', 'startFrame': 1, 'endFrame':474, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/container.txt', 'object_class': 'unknown' },
            { 'name':'couple', 'path':'data_seq/couple', 'startFrame': 1, 'endFrame':1826, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/couple.txt', 'object_class': 'unknown' },
            { 'name':'courier1', 'path':'data_seq/courier1', 'startFrame': 1, 'endFrame':703, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/courier1.txt', 'object_class': 'unknown' },
            { 'name':'courier2', 'path':'data_seq/courier2', 'startFrame': 1, 'endFrame':1705, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/courier2.txt', 'object_class': 'unknown' },
            { 'name':'dark car1-n', 'path':'data_seq/dark car1-n', 'startFrame': 1, 'endFrame':577, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/dark car1-n.txt', 'object_class': 'unknown' },
            { 'name':'dark car2-n', 'path':'data_seq/dark car2-n', 'startFrame': 1, 'endFrame':939, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/dark car2-n.txt', 'object_class': 'unknown' },
            { 'name':'duck1_1', 'path':'data_seq/duck1_1', 'startFrame': 1, 'endFrame':2175, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/duck1_1.txt', 'object_class': 'unknown' },
            { 'name':'duck1_2', 'path':'data_seq/duck1_2', 'startFrame': 1, 'endFrame':1551, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/duck1_2.txt', 'object_class': 'unknown' },
            { 'name':'duck2', 'path':'data_seq/duck2', 'startFrame': 1, 'endFrame':1131, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/duck2.txt', 'object_class': 'unknown' },
            { 'name':'duck3', 'path':'data_seq/duck3', 'startFrame': 1, 'endFrame':901, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/duck3.txt', 'object_class': 'unknown' },
            { 'name':'electric box', 'path':'data_seq/electric box', 'startFrame': 1, 'endFrame':1164, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/electric box.txt', 'object_class': 'unknown' },
            { 'name':'excavator', 'path':'data_seq/excavator', 'startFrame': 1, 'endFrame':630, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/excavator.txt', 'object_class': 'unknown' },
            { 'name':'football player1_1', 'path':'data_seq/football player1_1', 'startFrame': 1, 'endFrame':519, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/football player1_1.txt', 'object_class': 'unknown' },
            { 'name':'football player1_2', 'path':'data_seq/football player1_2', 'startFrame': 1, 'endFrame':1202, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/football player1_2.txt', 'object_class': 'unknown' },
            { 'name':'football player1_3', 'path':'data_seq/football player1_3', 'startFrame': 1, 'endFrame':1003, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/football player1_3.txt', 'object_class': 'unknown' },
            { 'name':'football player2_1', 'path':'data_seq/football player2_1', 'startFrame': 1, 'endFrame':546, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/football player2_1.txt', 'object_class': 'unknown' },
            { 'name':'football player2_2', 'path':'data_seq/football player2_2', 'startFrame': 1, 'endFrame':1173, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/football player2_2.txt', 'object_class': 'unknown' },
            { 'name':'group1', 'path':'data_seq/group1', 'startFrame': 1, 'endFrame':1321, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/group1.txt', 'object_class': 'unknown' },
            { 'name':'group2', 'path':'data_seq/group2', 'startFrame': 1, 'endFrame':1534, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/group2.txt', 'object_class': 'unknown' },
            { 'name':'group3_1', 'path':'data_seq/group3_1', 'startFrame': 1, 'endFrame':163, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/group3_1.txt', 'object_class': 'unknown' },
            { 'name':'group3_2', 'path':'data_seq/group3_2', 'startFrame': 1, 'endFrame':250, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/group3_2.txt', 'object_class': 'unknown' },
            { 'name':'group3_3', 'path':'data_seq/group3_3', 'startFrame': 1, 'endFrame':300, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/group3_3.txt', 'object_class': 'unknown' },
            { 'name':'group4', 'path':'data_seq/group4', 'startFrame': 1, 'endFrame':329, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/group4.txt', 'object_class': 'unknown' },
            { 'name':'group4_1', 'path':'data_seq/group4_1', 'startFrame': 1, 'endFrame':299, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/group4_1.txt', 'object_class': 'unknown' },
            { 'name':'group4_2', 'path':'data_seq/group4_2', 'startFrame': 1, 'endFrame':175, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/group4_2.txt', 'object_class': 'unknown' },
            { 'name':'hiker1', 'path':'data_seq/hiker1', 'startFrame': 1, 'endFrame':486, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/hiker1.txt', 'object_class': 'unknown' },
            { 'name':'hiker2', 'path':'data_seq/hiker2', 'startFrame': 1, 'endFrame':420, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/hiker2.txt', 'object_class': 'unknown' },
            { 'name':'human', 'path':'data_seq/human', 'startFrame': 1, 'endFrame':1605, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/human.txt', 'object_class': 'unknown' },
            { 'name':'human1', 'path':'data_seq/human1', 'startFrame': 1, 'endFrame':90, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/human1.txt', 'object_class': 'unknown' },
            { 'name':'human2', 'path':'data_seq/human2', 'startFrame': 1, 'endFrame':224, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/human2.txt', 'object_class': 'unknown' },
            { 'name':'human3', 'path':'data_seq/human3', 'startFrame': 1, 'endFrame':143, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/human3.txt', 'object_class': 'unknown' },
            { 'name':'human4', 'path':'data_seq/human4', 'startFrame': 1, 'endFrame':20, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/human4.txt', 'object_class': 'unknown' },
            { 'name':'human5', 'path':'data_seq/human5', 'startFrame': 1, 'endFrame':208, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/human5.txt', 'object_class': 'unknown' },
            { 'name':'island', 'path':'data_seq/island', 'startFrame': 1, 'endFrame':1969, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/island.txt', 'object_class': 'unknown' },
            { 'name':'jogging1', 'path':'data_seq/jogging1', 'startFrame': 1, 'endFrame':324, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/jogging1.txt', 'object_class': 'unknown' },
            { 'name':'jogging2', 'path':'data_seq/jogging2', 'startFrame': 1, 'endFrame':1717, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/jogging2.txt', 'object_class': 'unknown' },
            { 'name':'motor1', 'path':'data_seq/motor1', 'startFrame': 1, 'endFrame':57, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/motor1.txt', 'object_class': 'unknown' },
            { 'name':'motor2', 'path':'data_seq/motor2', 'startFrame': 1, 'endFrame':53, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/motor2.txt', 'object_class': 'unknown' },
            { 'name':'parterre1', 'path':'data_seq/parterre1', 'startFrame': 1, 'endFrame':2400, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/parterre1.txt', 'object_class': 'unknown' },
            { 'name':'parterre2', 'path':'data_seq/parterre2', 'startFrame': 1, 'endFrame':1705, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/parterre2.txt', 'object_class': 'unknown' },
            { 'name':'pot bunker', 'path':'data_seq/pot bunker', 'startFrame': 1, 'endFrame':1383, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/pot bunker.txt', 'object_class': 'unknown' },
            { 'name':'runner1', 'path':'data_seq/runner1', 'startFrame': 1, 'endFrame':333, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/runner1.txt', 'object_class': 'unknown' },
            { 'name':'runner2', 'path':'data_seq/runner2', 'startFrame': 1, 'endFrame':147, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/runner2.txt', 'object_class': 'unknown' },
            { 'name':'sand truck-n', 'path':'data_seq/sand truck-n', 'startFrame': 1, 'endFrame':829, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/sand truck-n.txt', 'object_class': 'unknown' },
            { 'name':'swan', 'path':'data_seq/swan', 'startFrame': 1, 'endFrame':300, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/swan.txt', 'object_class': 'unknown' },
            { 'name':'tennis player1_1', 'path':'data_seq/tennis player1_1', 'startFrame': 1, 'endFrame':850, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/tennis player1_1.txt', 'object_class': 'unknown' },
            { 'name':'tennis player1_2', 'path':'data_seq/tennis player1_2', 'startFrame': 1, 'endFrame':1655, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/tennis player1_2.txt', 'object_class': 'unknown' },
            { 'name':'tower crane', 'path':'data_seq/tower crane', 'startFrame': 1, 'endFrame':866, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/tower crane.txt', 'object_class': 'unknown' },
            { 'name':'tree', 'path':'data_seq/tree', 'startFrame': 1, 'endFrame':520, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/tree.txt', 'object_class': 'unknown' },
            { 'name':'tricycle1_1', 'path':'data_seq/tricycle1_1', 'startFrame': 1, 'endFrame':1102, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/tricycle1_1.txt', 'object_class': 'unknown' },
            { 'name':'tricycle1_2', 'path':'data_seq/tricycle1_2', 'startFrame': 1, 'endFrame':1113, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/tricycle1_2.txt', 'object_class': 'unknown' },
            { 'name':'truck', 'path':'data_seq/truck', 'startFrame': 1, 'endFrame':1871, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/truck.txt', 'object_class': 'unknown' },
            { 'name':'truck_night', 'path':'data_seq/truck_night', 'startFrame': 1, 'endFrame':1032, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/truck_night.txt', 'object_class': 'unknown' },
            { 'name':'uav1', 'path':'data_seq/uav1', 'startFrame': 1, 'endFrame':751, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/uav1.txt', 'object_class': 'unknown' },
            { 'name':'uav2', 'path':'data_seq/uav2', 'startFrame': 1, 'endFrame':1520, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/uav2.txt', 'object_class': 'unknown' },
            { 'name':'uav3_1', 'path':'data_seq/uav3_1', 'startFrame': 1, 'endFrame':634, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/uav3_1.txt', 'object_class': 'unknown' },
            { 'name':'uav3_2', 'path':'data_seq/uav3_2', 'startFrame': 1, 'endFrame':1208, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/uav3_2.txt', 'object_class': 'unknown' },
            { 'name':'uav4', 'path':'data_seq/uav4', 'startFrame': 1, 'endFrame':1533, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/uav4.txt', 'object_class': 'unknown' },
            { 'name':'uav5', 'path':'data_seq/uav5', 'startFrame': 1, 'endFrame':426, 'nz':5, 'ext': 'jpg', 'anno_path': 'anno/uav5.txt', 'object_class': 'unknown' },
        ]
        return sequence_info_list