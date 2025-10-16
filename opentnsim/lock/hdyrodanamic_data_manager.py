# config.py
class HydrodynamicDataManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HydrodynamicDataManager, cls).__new__(cls)
            cls._instance.hydrodynamic_data = None
            cls._instance.hydrodynamic_times = None

        return cls._instance
