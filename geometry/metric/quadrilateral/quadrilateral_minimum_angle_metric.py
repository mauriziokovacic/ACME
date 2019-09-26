from .quadrilateral_metric import *


class QuadrilateralMinimumAngleMetric(QuadrilateralMetric):
    def __init__(self):
        super(QuadrilateralMinimumAngleMetric, self).__init__(
            name='Quadrilateral Minimum Angle',
            dimension='A^1',
            acceptable_range=Range(min=45.0, max=90.0),
            normal_range=Range(min=0.0, max=90.0),
            full_range=Range(min=0.0, max=360.0),
            q_for_unit=90.0,
        )
