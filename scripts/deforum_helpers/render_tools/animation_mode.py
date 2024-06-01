
class AnimationMode:
    def __init__(self, filename, prev_flow):
        self._filename = filename
        self._prev_flow = prev_flow

    @property
    def filename(self):
        return self._filename

    @property
    def prev_flow(self):
        return self._prev_flow
