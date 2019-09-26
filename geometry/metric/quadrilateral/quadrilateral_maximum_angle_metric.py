from .quadrilateral_metric import *


class QuadrilateralMaximumAngleMetric(QuadrilateralMetric):
    def __init__(self):
        super(QuadrilateralMaximumAngleMetric, self).__init__(
            name='Quadrilateral Maximum Angle',
            dimension='1',
            acceptable_range=Range(min=1, max=1.3),
            normal_range=Range(min=1, max=Inf),
            full_range=Range(min=1, max=Inf),
            q_for_unit=1,
        )

