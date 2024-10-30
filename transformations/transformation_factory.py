
class TransformationsFactory:

    __transformations = {
        'Traslacion': Traslation(),
        'Rotacion': Rotation(),
        'Escalado': Scalated()
    }


    @stathic
    def initialize_transformations