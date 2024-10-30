
class TransformationsFactory:

    __transformations = {
        'Traslacion': Traslation(),
        'Rotacion': Rotation(),
        'Escalado': Scalated()
    }


    @staticmethod
    def initialize_transformation(key):
        return TransformationsFactory.__transformations[key]