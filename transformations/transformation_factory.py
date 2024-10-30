from transformations.traslation import Traslation
from transformations.rotation import Rotation
from transformations.scalated import Scalated

class TransformationsFactory:

    __transformations = {
        'Traslacion': Traslation(),
        'Rotacion': Rotation(),
        'Escalado': Scalated()
    }


    @staticmethod
    def initialize_transformation(key):
        return TransformationsFactory.__transformations[key]