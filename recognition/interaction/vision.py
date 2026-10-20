from pathlib import Path
from typing import Optional, Dict, Any, Callable, List

from recognition.core.inference.furhat_service import FurhatIdentitySystem
from recognition.core.utils.database import IdentityClass


class Vision(FurhatIdentitySystem):
    def __init__(
            self,
            model_name: str,
            weight_path: Path | str,
            database: List[IdentityClass],
            use_webcam=False,
            callback: Optional[Callable[[Dict[str, Any]], None]] = None,
            furhat_client=None,
            similarity_threshold=float,

    ):
        super().__init__(
            callback=callback,
            use_webcam=use_webcam,
            furhat_client=furhat_client,
            similarity_threshold=similarity_threshold,
            model_name=model_name,
            model_path=weight_path,
            database=database
        )

    async def start_monitoring(self):
        """Starts monitoring and continuously dispatches callback events."""
        await self.run()
