from collections import Counter

class CompleteObserver:
    """Has access to all the information about the sequence, returns the full ratio of colors in each quadrant"""
    
    def __init__(self, seq_gen):
        self.seq_gen = seq_gen
        self.point_data = []
        self.quad_data = []
        self.ratios = None
        self.counts = None

    def quad_identifier(self, position):
        h = self.seq_gen.height // 2
        w = self.seq_gen.width // 2
        if position[0] < h and position[1] < w:
            return "TL"
        elif position[0] < h and position[1] >= w:
            return "TR"
        elif position[0] >= h and position[1] < w:
            return "BL"
        else:
            return "BR"

    def process_sequence(self, sequence):
        self.point_data = [(e.position, e.true_color) for s in sequence for e in s]
        self.quad_data = [(self.quad_identifier(e[0]), e[1].name) for e in self.point_data]
        self._calculate_stats()
        return self.ratios

    def _calculate_stats(self):
        positions = {'TL': {'RED': 0, 'GREEN': 0},
                    'TR': {'RED': 0, 'GREEN': 0},
                    'BL': {'RED': 0, 'GREEN': 0},
                    'BR': {'RED': 0, 'GREEN': 0}}
        
        for pos, color in self.quad_data:
            positions[pos][color] += 1
            
        for pos in positions:
            total = sum(positions[pos].values())
            if total:
                for color in positions[pos]:
                    positions[pos][color] = positions[pos][color] / total
                    
        self.ratios = positions
        self.counts = Counter(self.quad_data)

