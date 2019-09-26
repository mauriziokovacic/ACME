from .quadrilateral_metric import *


class QuadrilateralOddyMetric(QuadrilateralMetric):
    def __init__(self):
        super(QuadrilateralOddyMetric, self).__init__(
            name='Quadrilateral Oddy',
            dimension='1',
            acceptable_range=Range(min=0, max=0.5),
            normal_range=Range(min=0, max=Inf),
            full_range=Range(min=0, max=Inf),
            q_for_unit=0,
        )
