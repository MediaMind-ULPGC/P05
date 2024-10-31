from transformations.traslation import Traslation
from transformations.rotation import Rotation
from transformations.scalated import Scalated
from transformations.distortions import Distortions
from transformations.apply_noise import ApplyNoise

class TransformationsFactory:

    __transformations = {
        'Traslacion': Traslation(),
        'Rotacion': Rotation(),
        'Escalado': Scalated(),
        'Distorciones': Distortions(),
        'Ruido': ApplyNoise()
    }


    @staticmethod
    def initialize_transformation(key):
        return TransformationsFactory.__transformations[key]