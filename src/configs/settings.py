import os
import json
from typing import Dict, Any, Optional


class Settings:
    DEFAULT_CONFIG = {
        "blur": {"type": "Gaussian Blur", "ksize": 9},
        "threshold": {
            "adaptive_type": "Gaussian",
            "thresh_type": "Binary",
            "block_size": 125,
            "c_index": 9,
        },
        "morphological": {"type": "Erode", "kernel_size": 5},
        "contour": {"retrieval_mode": "EXTERNAL", "approximation_mode": "SIMPLE"},
        "detection": {"area_min": "100000", "area_max": "150000", "distance": 15},
    }

    UI_PARAMETERS = {
        "blur": {
            "types": ["Gaussian Blur", "Median Blur", "Average Blur"],
            "ksize_range": (1, 31),
            "ksize_step": 2,
        },
        "threshold": {
            "adaptive_types": ["Gaussian", "Mean"],
            "thresh_types": [
                "Binary",
                "Binary Inverted",
                "Truncate",
                "To Zero",
                "To Zero Inverted",
            ],
            "block_size_range": (3, 255),
            "block_size_step": 2,
            "c_index_range": (-255, 255),
        },
        "morphological": {
            "types": ["Erode", "Dilate", "Open", "Close"],
            "kernel_range": (1, 31),
            "kernel_step": 2,
        },
        "contour": {
            "retrieval_modes": ["EXTERNAL", "LIST", "CCOMP", "TREE"],
            "approximation_modes": ["NONE", "SIMPLE", "TC89_L1", "TC89_KCOS"],
        },
        "detection": {
            "distance_range": (0, 100),
        },
    }

    def __init__(self):
        self.models_dir = os.path.join("src", "models")
        os.makedirs(self.models_dir, exist_ok=True)

    def get_ui_parameter(self):
        return self.UI_PARAMETERS

    def get_default_config(self):
        return self.DEFAULT_CONFIG

    def get_model_names(self) -> list:
        """Get list of available model names"""
        if not os.path.exists(self.models_dir):
            print("not os.path.exists(self.models_dir)")
            return []

        models = [
            f
            for f in os.listdir(self.models_dir)
            if os.path.isdir(os.path.join(self.models_dir, f))
        ]

        return models

    def get_model_path(self, model_name: str) -> str:
        """Get full path for a model's config file"""
        return os.path.join(self.models_dir, model_name, "config.json")

    def load_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Load configuration for specified model"""
        try:
            config_path = self.get_model_path(model_name)
            if not os.path.exists(config_path):
                return None

            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            return None

    def save_model(self, model_name: str, config: Dict[str, Any]) -> bool:
        """Save configuration for specified model"""
        try:
            # Create model directory if it doesn't exist
            model_dir = os.path.join(self.models_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)

            # Save configuration
            config_path = self.get_model_path(model_name)
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving model {model_name}: {str(e)}")
            return False

    def delete_model(self, model_name: str) -> bool:
        """Delete specified model and its configuration"""
        try:
            model_dir = os.path.join(self.models_dir, model_name)
            if os.path.exists(model_dir):
                import shutil

                shutil.rmtree(model_dir)
            return True
        except Exception as e:
            print(f"Error deleting model {model_name}: {str(e)}")
            return False

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """Validate configuration structure"""
        try:
            required_sections = [
                "blur",
                "threshold",
                "morphological",
                "contour",
                "detection",
            ]
            for section in required_sections:
                if section not in config:
                    return False

            # Add specific validation for each section if needed
            return True
        except Exception:
            return False
