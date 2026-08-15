"""VAMInference + the ablated (no world-model) pipeline. Subclasses
VAMInference unchanged (video_action_model.py) -- constructs it normally
(is_hil=False, so it loads video2world_pipeline/world2action_pipeline as
usual, no wasted work), then swaps self.model for ZeroWorldModelPipeline,
reusing the same already-loaded sub-pipelines rather than reloading
weights. reset()/step() are inherited unmodified.
"""

from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, "/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/SimplerEnv")

from simpler_env.policies.vam.video_action_model import VAMInference  # noqa: E402

from zero_world_model_pipeline import ZeroWorldModelPipeline  # noqa: E402


class VAMAblatedInference(VAMInference):
    def __init__(self, *args, **kwargs):
        kwargs["is_hil"] = False
        super().__init__(*args, **kwargs)
        self.model = ZeroWorldModelPipeline(
            self.model.video2world_pipeline, self.model.world2action_pipeline
        ).cuda()
