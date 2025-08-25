"""
Enhanced Component Confidence for test compatibility
"""

from typing import Optional


class ComponentConfidenceTest:
    """Component confidence compatible with test interface"""

    def __init__(
        self,
        il: float = 0.0,
        ilce: float = 0.0,
        mahalle: float = 0.0,
        yol: float = 0.0,
        sokak: float = 0.0,
        bina_no: float = 0.0,
        daire_no: float = 0.0,
        value: float = 0.85,
        source: str = "test",
    ):
        """Initialize with component confidences"""
        self.il = il
        self.ilce = ilce
        self.mahalle = mahalle
        self.yol = yol
        self.sokak = sokak
        self.bina_no = bina_no
        self.daire_no = daire_no
        self.value = value if value != 0.85 else (il + ilce + mahalle + yol + bina_no + daire_no) / 6
        self.source = source
        self.method = "test"
