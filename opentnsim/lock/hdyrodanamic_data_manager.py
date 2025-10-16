# config.py
class HydrodynamicDataManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HydrodynamicDataManager, cls).__new__(cls)
            cls._instance.hydrodynamic_data = None
        return cls._instance



# main.py

manager1 = HydrodynamicDataManager()
manager2 = HydrodynamicDataManager()

print(manager1.hydrodynamic_data)  # Output: None
manager2.hydrodynamic_data = "Some data"
print(manager1.hydrodynamic_data)  # Output: "Some data" (Same instance)