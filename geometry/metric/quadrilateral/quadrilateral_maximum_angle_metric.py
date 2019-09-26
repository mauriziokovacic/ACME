from .quadrilateral_metric import *


class QuadrilateralMaximumAngleMetric(QuadrilateralMetric):
    def __init__(self):
        super(QuadrilateralMaximumAngleMetric, self).__init__(
            name='Quadrilateral Maximum Angle',
            dimension='A^1',
            acceptable_range=Range(min=90.0, max=135.0),
            normal_range=Range(min=90.0, max=360.0),
            full_range=Range(min=0.0, max=360.0),
            q_for_unit=90.0,
        )

