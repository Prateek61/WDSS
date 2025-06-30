import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
from typing import Dict, List, Tuple

class NetworkLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.writer = SummaryWriter(log_path)
        self.writers: Dict[str, SummaryWriter] = {}
        
        # Cache for scalar and image data
        self.cached_scalars = {}
        self.cached_images = {}

    def log_scalar(self, tag: str, scalar_value: float, global_step: int, component_name: str = "") -> None:
        self.get_writer(component_name).add_scalar(tag, scalar_value, global_step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], global_step: int, component_name: str = "") -> None:
        self.get_writer(component_name).add_scalars(main_tag, tag_scalar_dict, global_step)

    def log_image(self, tag: str, img_tensor, global_step: int, component_name: str = "") -> None:
        self.get_writer(component_name).add_image(tag, img_tensor, global_step)

    def get_scalars_from_path(self, path: str, file_name: str) -> Dict[str, List[Tuple[int, float]]]:
        full_path = os.path.join(path, file_name)
        if full_path in self.cached_scalars:
            return self.cached_scalars[full_path]
        
        event_acc = EventAccumulator(full_path)
        event_acc.Reload()
        scalars = self._get_scalars(event_acc)
        self.cached_scalars[full_path] = scalars
        return scalars

    def get_all_scalars(self, file_name: str) -> Dict[str, Dict[str, List[Tuple[int, float]]]]:
        all_scalars = {}
        full_path = os.path.join(self.log_path, file_name)
        for files in os.listdir(full_path):
            if files.startswith("events"):
                event_acc = EventAccumulator(os.path.join(full_path, files))
                event_acc.Reload()
                scalars = self._get_scalars(event_acc)
                for tag, data in scalars.items():
                    if tag not in all_scalars:
                        all_scalars[tag] = []
                    all_scalars[tag].extend(data)
        return all_scalars
    
    def get_image_tags(self, path: str) -> List[str]:
        if path not in self.cached_images:
            event_acc = EventAccumulator(path, size_guidance={'images': 0})
            event_acc.Reload()
            self.cached_images[path] = self._get_image_tags(event_acc)
        return self.cached_images[path]
    
    def get_images_by_tag(self, path: str, tag: str) -> List[Tuple[int, bytes]]:
        cache_key = f"{path}_{tag}"
        if cache_key in self.cached_images:
            return self.cached_images[cache_key]
        
        event_acc = EventAccumulator(path, size_guidance={'images': 0})
        event_acc.Reload()
        images = self._get_images_by_tag(event_acc, tag)
        self.cached_images[cache_key] = images
        return images

    def _get_scalars(self, event_acc: EventAccumulator) -> Dict[str, List[Tuple[int, float]]]:
        tags = event_acc.Tags()['scalars']
        data = {}
        for tag in tags:
            data[tag] = [(s.step, s.value) for s in event_acc.Scalars(tag)]        
        return data
    
    def _get_image_tags(self, event_acc: EventAccumulator) -> List[str]:
        return event_acc.Tags().get("images", [])
    
    def _get_images_by_tag(self, event_acc: EventAccumulator, tag: str) -> List[Tuple[int, bytes]]:
        image_tags = event_acc.Tags().get("images", [])
        if tag not in image_tags:
            print(f"Tag '{tag}' not found in logs.")
            return []
        image_events = event_acc.Images(tag)
        return [(img.step, img.encoded_image_string) for img in image_events]

    def get_writer(self, component_name: str = "") -> SummaryWriter:
        if component_name == "":
            return self.writer

        if component_name not in self.writers:
            path = os.path.join(self.log_path, component_name)
            os.makedirs(path, exist_ok=True)
            self.writers[component_name] = SummaryWriter(path)

        return self.writers[component_name]